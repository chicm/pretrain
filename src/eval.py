"""Evaluate a Chimera pretraining checkpoint with EleutherAI lm-evaluation-harness.

Base (pretrained, no instruction-tuning) models are scored by LOG-LIKELIHOOD, not
generation, so all the tasks here work on a raw checkpoint:

  * hellaswag        - commonsense sentence completion (4-way, acc / acc_norm)
  * lambada_openai   - last-word prediction over a long context (acc, perplexity)
  * arc_easy         - grade-school science MC
  * arc_challenge    - harder science MC
  * piqa, winogrande - optional extra commonsense (add via --tasks)
  * mmlu             - 57-subject knowledge (OFF by default; enable with --mmlu).
                       NOTE: a 1B model at ~2B tokens will sit near the 25% random
                       baseline. MMLU only becomes meaningful for >=7B / >=few-100B
                       tokens. Left off by default on purpose.

Usage (run on a GPU node, inside the py_3.10 conda env):

    pip install "lm-eval>=0.4.3"          # one-time, --user is fine on shared env
    python eval.py --ckpt $OUT/ckpt_2000.pt \
        --tasks hellaswag,lambada_openai,arc_easy,arc_challenge \
        --batch_size 16 --output_dir $OUT/eval

    # include MMLU (slow, and expect ~random for small/early models):
    python eval.py --ckpt ... --mmlu

Single-GPU by design (eval is cheap vs. training); pick the GPU with
CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES.
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

from model import Chimera, ModelArgs
from configs import MODELS, VOCAB_SIZE

try:
    from transformers import AutoTokenizer
except Exception as e:  # pragma: no cover
    print("transformers is required:", e)
    raise

# lm-eval is imported lazily in main() so `python eval.py --help` works without it.

DEFAULT_TASKS = "hellaswag,lambada_openai,arc_easy,arc_challenge"
MMLU_TASK = "mmlu"                      # lm-eval group covering all 57 subjects
DEFAULT_TOKENIZER = "Qwen/Qwen3-8B"     # must match tokenize_data.py


# --------------------------------------------------------------------------- #
# checkpoint loading
# --------------------------------------------------------------------------- #
def build_model_args(cfg: dict) -> ModelArgs:
    """Rebuild ModelArgs from the checkpoint's saved cfg dict.

    train.py stores cfg.__dict__, whose 'model' field is a preset name
    ("tiny"/"1b"/"8b"). We rebuild from the preset so architecture always
    matches how the checkpoint was trained, then sanity-check vocab.
    """
    preset = cfg.get("model", "1b")
    if preset not in MODELS:
        raise ValueError(f"unknown model preset in ckpt cfg: {preset!r}")
    margs = MODELS[preset]()
    # block_size may have been overridden at train time; keep model's own
    # max_seq_len (that's what RoPE tables were built for).
    return margs


def load_checkpoint(path: str, device: str):
    print(f"[eval] loading checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg", {})
    step = ckpt.get("step", "?")
    margs = build_model_args(cfg)
    print(f"[eval] preset={cfg.get('model','?')} step={step} "
          f"dim={margs.dim} n_layers={margs.n_layers} vocab={margs.vocab_size} "
          f"max_seq_len={margs.max_seq_len}")

    model = Chimera(margs)
    sd = ckpt["model"]
    # strip any compile/DDP prefixes just in case
    sd = { k.replace("_orig_mod.", "").replace("module.", ""): v
           for k, v in sd.items() }
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # tied lm_head.weight may be absent from a saved sd (shares tok_emb) -> fine.
    missing = [m for m in missing if m not in ("lm_head.weight",)]
    if missing:
        print(f"[eval][warn] missing keys: {missing[:8]}"
              f"{' ...' if len(missing) > 8 else ''}")
    if unexpected:
        print(f"[eval][warn] unexpected keys: {unexpected[:8]}"
              f"{' ...' if len(unexpected) > 8 else ''}")

    model.to(device).eval()
    return model, margs, cfg, step


# --------------------------------------------------------------------------- #
# lm-eval adapter
# --------------------------------------------------------------------------- #
def make_lm_class():
    """Build the ChimeraLM class (defined inside so lm_eval import is lazy)."""
    from lm_eval.api.model import LM
    from lm_eval.api.registry import register_model

    class ChimeraLM(LM):
        """Minimal lm-eval wrapper around a Chimera checkpoint.

        Implements loglikelihood (multiple-choice tasks) and
        loglikelihood_rolling (perplexity tasks). generate_until is not
        implemented because none of the base-model tasks here need free
        generation.
        """

        def __init__(self, model, tokenizer, max_length, device,
                     batch_size=16, dtype=torch.bfloat16):
            super().__init__()
            self.model = model
            self.tok = tokenizer
            self._max_length = int(max_length)
            self._device = device
            self.batch_size = int(batch_size)
            self.dtype = dtype
            # Qwen3 has no dedicated BOS; use <|endoftext|> as the prefix token
            # when a request has empty context.
            self.prefix_id = self.tok.eos_token_id
            if self.prefix_id is None:
                self.prefix_id = self.tok.convert_tokens_to_ids("<|endoftext|>")

        # -- tokenization helpers -----------------------------------------
        def tok_encode(self, string, **kwargs):
            return self.tok.encode(string, add_special_tokens=False)

        def tok_decode(self, tokens, **kwargs):
            return self.tok.decode(tokens)

        @property
        def eot_token_id(self):
            return self.prefix_id

        @property
        def max_length(self):
            return self._max_length

        # -- core scoring -------------------------------------------------
        @torch.no_grad()
        def _score_batch(self, batch):
            """batch: list of (input_ids(list), cont_len(int)).
            Returns list of (sum_logprob(float), is_greedy(bool)).
            Right-pads to a common length; causal model => padding at the end
            does not affect earlier positions, and we only read real positions.
            """
            maxlen = max(len(ids) for ids, _ in batch)
            input_ids = torch.full((len(batch), maxlen), self.prefix_id,
                                   dtype=torch.long, device=self._device)
            for i, (ids, _) in enumerate(batch):
                input_ids[i, : len(ids)] = torch.tensor(ids, device=self._device)

            with torch.autocast(device_type=self._device.split(":")[0],
                                dtype=self.dtype):
                logits, _ = self.model(input_ids)
            logits = logits.float()
            logprobs = F.log_softmax(logits, dim=-1)

            out = []
            for i, (ids, cont_len) in enumerate(batch):
                seq_len = len(ids)
                # continuation occupies the last cont_len tokens of ids;
                # they are predicted by positions [seq_len-cont_len-1, seq_len-2].
                start = seq_len - cont_len
                # logits at position t predict token t+1
                pred_slice = logprobs[i, start - 1: seq_len - 1, :]   # (cont_len, V)
                target = torch.tensor(ids[start:seq_len], device=self._device)
                tok_lp = pred_slice.gather(1, target.unsqueeze(1)).squeeze(1)
                greedy = (pred_slice.argmax(-1) == target).all().item()
                out.append((tok_lp.sum().item(), bool(greedy)))
            return out

        def _encode_pair(self, context, continuation):
            if context == "":
                ctx_ids = [self.prefix_id]
            else:
                ctx_ids = self.tok_encode(context)
            cont_ids = self.tok_encode(continuation)
            ids = ctx_ids + cont_ids
            # left-truncate to max_length, but never drop continuation tokens
            if len(ids) > self.max_length:
                overflow = len(ids) - self.max_length
                # keep at least the continuation; trim from the context side
                keep_ctx = max(0, len(ctx_ids) - overflow)
                ids = ctx_ids[len(ctx_ids) - keep_ctx:] + cont_ids
                ids = ids[-self.max_length:]
            return ids, len(cont_ids)

        def loglikelihood(self, requests, disable_tqdm=False):
            from tqdm import tqdm
            # build (ids, cont_len), remember order
            packed = []
            for req in requests:
                context, continuation = req.args
                packed.append(self._encode_pair(context, continuation))
            # sort by length for efficient batching, keep original index
            order = sorted(range(len(packed)), key=lambda i: len(packed[i][0]))
            results = [None] * len(packed)
            bs = self.batch_size
            for b in tqdm(range(0, len(order), bs),
                          disable=disable_tqdm, desc="loglikelihood"):
                idxs = order[b: b + bs]
                batch = [packed[i] for i in idxs]
                scored = self._score_batch(batch)
                for i, s in zip(idxs, scored):
                    results[i] = s
            return results

        def loglikelihood_rolling(self, requests, disable_tqdm=False):
            from tqdm import tqdm
            results = []
            for req in tqdm(requests, disable=disable_tqdm,
                            desc="loglikelihood_rolling"):
                (string,) = req.args
                token_ids = self.tok_encode(string)
                # sliding non-overlapping windows of max_length, score every
                # token given all previous (first token conditioned on prefix).
                ids = [self.prefix_id] + token_ids
                total = 0.0
                for s in range(0, len(token_ids), self.max_length - 1):
                    window = ids[s: s + self.max_length]
                    if len(window) < 2:
                        break
                    cont_len = len(window) - 1
                    scored = self._score_batch([(window, cont_len)])
                    total += scored[0][0]
                results.append((total,))
            return results

        def generate_until(self, requests, disable_tqdm=False):
            raise NotImplementedError(
                "generate_until is not needed for the base-model likelihood "
                "tasks; add a generation path if you want free-form eval.")

    return ChimeraLM


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="path to ckpt_STEP.pt")
    ap.add_argument("--tasks", default=DEFAULT_TASKS,
                    help=f"comma-separated lm-eval tasks (default: {DEFAULT_TASKS})")
    ap.add_argument("--mmlu", action="store_true",
                    help="also run MMLU (slow; ~random for small/early models)")
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--limit", type=float, default=None,
                    help="limit #examples per task (int) or fraction (0-1); for quick smoke")
    ap.add_argument("--num_fewshot", type=int, default=None,
                    help="override few-shot (default: task default; MMLU=5)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output_dir", default=None,
                    help="write results json here (default: <ckpt_dir>/eval)")
    args = ap.parse_args()

    tasks = [t for t in args.tasks.split(",") if t]
    if args.mmlu and MMLU_TASK not in tasks:
        tasks.append(MMLU_TASK)

    # tokenizer
    print(f"[eval] tokenizer: {args.tokenizer}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if len(tok) > VOCAB_SIZE:
        print(f"[eval][warn] tokenizer len {len(tok)} > model vocab {VOCAB_SIZE}")

    # model
    model, margs, cfg, step = load_checkpoint(args.ckpt, args.device)

    # lm-eval
    try:
        from lm_eval import evaluator
    except Exception as e:
        print("\n[eval] lm-eval not installed. Install with:\n"
              "    pip install --user 'lm-eval>=0.4.3'\n")
        raise

    ChimeraLM = make_lm_class()
    lm = ChimeraLM(model, tok, max_length=margs.max_seq_len,
                   device=args.device, batch_size=args.batch_size)

    print(f"[eval] running tasks: {tasks}")
    res = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        batch_size=args.batch_size,
        bootstrap_iters=1000,
    )

    # --- print + save ---
    table = res.get("results", {})
    print("\n================ RESULTS ================")
    for task, metrics in table.items():
        line = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                         if isinstance(v, (int, float)) and not k.endswith("_stderr"))
        print(f"  {task:24s} {line}")
    print("=========================================")

    out_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.ckpt)), "eval")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_name = os.path.splitext(os.path.basename(args.ckpt))[0]
    out_path = os.path.join(out_dir, f"eval_{ckpt_name}.json")
    with open(out_path, "w") as f:
        json.dump({"ckpt": args.ckpt, "step": step, "tasks": tasks,
                   "results": table, "config": {"model": cfg.get("model"),
                   "num_fewshot": args.num_fewshot, "limit": args.limit}},
                  f, indent=2)
    print(f"[eval] wrote {out_path}")


if __name__ == "__main__":
    main()
