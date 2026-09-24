"""
Step 1 - Embed every unique shadow record once with ClinicalBERT (PyTorch).

The k shadow models differ only in WHICH pairs they train on, not in how the
text is embedded, so the expensive BERT pass is done once here and cached.
Deduplicating by uid cuts it from 79,998 forward passes to 55,461.

Checkpoints every --ckpt-every batches, so a long CPU run survives a crash:
just re-run the same command and it picks up where it stopped.

Usage:
    python embed_shadow.py --pool shadow_clean/shadow_pool.csv --out-dir shadow_emb
    python embed_shadow.py --batch-size 64        # larger if you have GPU memory
    python embed_shadow.py --fp16                 # GPU only, roughly 2x faster
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel


def embed_texts(texts, tokenizer, model, device, batch_size=32, max_length=512,
                fp16=False, ckpt_path=None, ckpt_every=200, start_at=0, out=None):
    """CLS embeddings for `texts`, in the order given."""
    hidden = model.config.hidden_size
    if out is None:
        out = np.zeros((len(texts), hidden), dtype=np.float32)

    # Long texts first: the first batch is then the worst case, so an
    # out-of-memory blows up in the first 30 seconds instead of hour three.
    order = np.argsort([-len(t) for t in texts])

    t0, done = time.time(), start_at
    for b_start in range(start_at, len(texts), batch_size):
        idx = order[b_start:b_start + batch_size]
        batch = [texts[i] for i in idx]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            if fp16 and device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    # Pass the attention mask: without it padded positions feed
                    # into self-attention and the CLS vector shifts with batch makeup.
                    hs = model(**enc).last_hidden_state
            else:
                hs = model(**enc).last_hidden_state
        out[idx] = hs[:, 0, :].float().cpu().numpy()

        done = b_start + len(idx)
        n_batches = (b_start - start_at) // batch_size + 1
        if n_batches % 10 == 0 or done >= len(texts):
            rate = (done - start_at) / (time.time() - t0)
            eta = (len(texts) - done) / rate / 60 if rate else 0
            print(f"  {done}/{len(texts)}  {rate:.1f} rec/s  ETA {eta:.0f} min",
                  flush=True)
        if ckpt_path and n_batches % ckpt_every == 0:
            np.save(ckpt_path, out)
            with open(ckpt_path + ".done", "w") as f:
                f.write(str(done))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="shadow_clean/shadow_pool.csv")
    ap.add_argument("--out-dir", default="shadow_emb")
    ap.add_argument("--model", default="emilyalsentzer/Bio_ClinicalBERT")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--fp16", action="store_true", help="GPU only")
    ap.add_argument("--ckpt-every", type=int, default=200, help="batches between saves")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    emb_path = os.path.join(args.out_dir, "uid_embeddings.npy")
    uid_path = os.path.join(args.out_dir, "uid_index.npy")
    ckpt_path = os.path.join(args.out_dir, "_partial.npy")
    if os.path.exists(emb_path) and os.path.exists(uid_path):
        print(f"cache already present at {args.out_dir} - delete it to re-embed")
        return

    pool = pd.read_csv(args.pool)
    # uid <-> text is 1:1 in this dataset, so uid is a safe key for the cache.
    recs = pd.concat([
        pool[["uid1", "text1"]].rename(columns={"uid1": "uid", "text1": "text"}),
        pool[["uid2", "text2"]].rename(columns={"uid2": "uid", "text2": "text"}),
    ], ignore_index=True).drop_duplicates(subset="uid").reset_index(drop=True)
    texts = recs["text"].astype(str).tolist()
    print(f"{len(texts)} unique records to embed (from {2 * len(pool)} pair slots)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else
             f" ({os.cpu_count()} cores) - expect hours, consider Colab"))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()

    start_at, out = 0, None
    if os.path.exists(ckpt_path) and os.path.exists(ckpt_path + ".done"):
        out = np.load(ckpt_path)
        start_at = int(open(ckpt_path + ".done").read().strip())
        start_at -= start_at % args.batch_size          # redo the partial batch
        print(f"resuming from checkpoint at record {start_at}")

    out = embed_texts(texts, tokenizer, model, device,
                      batch_size=args.batch_size, max_length=args.max_length,
                      fp16=args.fp16, ckpt_path=ckpt_path,
                      ckpt_every=args.ckpt_every, start_at=start_at, out=out)

    np.save(emb_path, out)
    np.save(uid_path, recs["uid"].to_numpy().astype(str))
    for p in (ckpt_path, ckpt_path + ".done"):
        if os.path.exists(p):
            os.remove(p)
    print(f"wrote {emb_path} {out.shape} and {uid_path}")


if __name__ == "__main__":
    main()
