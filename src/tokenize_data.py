"""Tokenize a downloaded HF dataset into packed .bin. Run once before training.

Examples:
  # TinyStories smoke test (GPT2 tokenizer)
  python tokenize_data.py --dataset roneneldan/TinyStories --split train \
      --out ./data/tinystories_tok

  # FineWeb sample-10BT
  python tokenize_data.py --dataset HuggingFaceFW/fineweb --hf_config sample-10BT \
      --split train --out ./data/fineweb_tok
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
    ap.add_argument("--tokenizer", default="gpt2")
    ap.add_argument("--num_proc", type=int, default=32)
    ap.add_argument("--streaming", action="store_true")
    ap.add_argument("--max_docs", type=int, default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if tok.eos_token_id is None:
        tok.eos_token = tok.pad_token or "<|endoftext|>"
    prepare_data(args.dataset, args.out, tok, split=args.split,
                 text_key=args.text_key, num_proc=args.num_proc,
                 hf_config=args.hf_config, streaming=args.streaming,
                 max_docs=args.max_docs)


if __name__ == "__main__":
    main()
