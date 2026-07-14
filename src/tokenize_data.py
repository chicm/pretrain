"""Tokenize a downloaded HF dataset into packed .bin. Run once before training.

Examples:
  # TinyStories smoke test
  python tokenize_data.py --dataset roneneldan/TinyStories --split train \
      --out ./data/tinystories_tok

  # FineWeb sample-10BT (Qwen3 tokenizer, the default)
  python tokenize_data.py --dataset HuggingFaceFW/fineweb --hf_config sample-10BT \
      --split train --out ./data/fineweb_tok --num_proc 64
"""
import argparse
from transformers import AutoTokenizer
from data import prepare_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--hf_config", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--text_key", default="text")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-8B",
                    help="HF tokenizer name. Default: Qwen3 (vocab 151669).")
    ap.add_argument("--eot_token", default="<|endoftext|>",
                    help="Document separator token appended after each doc. "
                         "Qwen3 pretraining uses <|endoftext|> (id 151643), "
                         "NOT the chat <|im_end|>.")
    ap.add_argument("--num_proc", type=int, default=32)
    ap.add_argument("--streaming", action="store_true")
    ap.add_argument("--max_docs", type=int, default=None)
    ap.add_argument("--sharded", action="store_true",
                    help="streaming resumable sharded output (for 1T sources)")
    ap.add_argument("--shard_tokens", type=int, default=1_000_000_000,
                    help="tokens per shard (default 1B)")
    ap.add_argument("--target_tokens", type=int, default=None,
                    help="stop after this many tokens (per-source budget)")
    ap.add_argument("--data_files", default=None,
                    help="optional HF data_files glob/pattern")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    # resolve the document-separator token id explicitly
    eot_id = tok.convert_tokens_to_ids(args.eot_token)
    if eot_id is None or eot_id == tok.unk_token_id:
        eot_id = tok.eos_token_id if tok.eos_token_id is not None else 0
    print(f"[tokenize] tokenizer={args.tokenizer} vocab={tok.vocab_size} "
          f"eot='{args.eot_token}' id={eot_id}")
    if args.sharded:
        from data import prepare_data_sharded
        prepare_data_sharded(
            args.dataset, args.out, tok, split=args.split,
            text_key=args.text_key, hf_config=args.hf_config, eot_id=eot_id,
            shard_tokens=args.shard_tokens, target_tokens=args.target_tokens,
            num_proc=args.num_proc, data_files=args.data_files, streaming=True)
    else:
        prepare_data(args.dataset, args.out, tok, split=args.split,
                     text_key=args.text_key, num_proc=args.num_proc,
                     hf_config=args.hf_config, streaming=args.streaming,
                     max_docs=args.max_docs, eot_id=eot_id)


if __name__ == "__main__":
    main()
