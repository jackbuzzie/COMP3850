"""
Step 0 - Clean the shadow CSVs.

shadow_test.csv arrives with 238 trailing unnamed columns, mixed-type labels
('1' as string vs 1 as int) and at least one row where a record text has
spilled into the label column. Training k shadow models on that as-is would
silently drop or mislabel pairs, so we repair it once and write clean copies.

Usage:
    python clean_shadow_data.py --train shadow_train.csv --test shadow_test.csv --out-dir shadow_clean
"""
import argparse
import os
import numpy as np
import pandas as pd

KEEP = ["uid1", "text1", "uid2", "text2", "label"]


def clean(path, name):
    df = pd.read_csv(path, low_memory=False)
    report = {"file": name, "rows_read": len(df), "cols_read": df.shape[1]}

    extra = [c for c in df.columns if str(c).startswith("Unnamed")]
    report["trailing_cols_dropped"] = len(extra)
    missing = [c for c in KEEP if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing required columns {missing}")
    df = df[KEEP].copy()

    # Label must be exactly 0 or 1. Anything else (a spilled text fragment,
    # an empty cell) marks a row whose fields have shifted -> drop it.
    lab = pd.to_numeric(df["label"], errors="coerce")
    bad = ~lab.isin([0, 1])
    report["bad_label_rows_dropped"] = int(bad.sum())
    df, lab = df[~bad], lab[~bad]
    df["label"] = lab.astype(np.int64)

    # Any remaining null in a field we actually use.
    null_mask = df[KEEP].isna().any(axis=1)
    report["null_rows_dropped"] = int(null_mask.sum())
    df = df[~null_mask]

    for c in ("text1", "text2", "uid1", "uid2"):
        df[c] = df[c].astype(str).str.strip()

    empty = (df["text1"].str.len() == 0) | (df["text2"].str.len() == 0)
    report["empty_text_rows_dropped"] = int(empty.sum())
    df = df[~empty]

    dup = df.duplicated(subset=["uid1", "uid2", "label"])
    report["duplicate_pairs_dropped"] = int(dup.sum())
    df = df[~dup].reset_index(drop=True)

    report["rows_kept"] = len(df)
    report["label_1"] = int((df["label"] == 1).sum())
    report["label_0"] = int((df["label"] == 0).sum())
    return df, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out-dir", default="shadow_clean")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tr, rep_tr = clean(args.train, "shadow_train")
    te, rep_te = clean(args.test, "shadow_test")

    for rep in (rep_tr, rep_te):
        print(f"--- {rep['file']} ---")
        for k, v in rep.items():
            if k != "file":
                print(f"  {k:28s} {v}")

    # The shadow pool is the union: which side of the original split a pair
    # came from is irrelevant once WE decide member/non-member per shadow model.
    tr["orig_split"] = "train"
    te["orig_split"] = "test"
    pool = pd.concat([tr, te], ignore_index=True)

    dup_across = pool.duplicated(subset=["uid1", "uid2", "label"])
    print(f"\npairs duplicated across the two files: {int(dup_across.sum())}")
    pool = pool[~dup_across].reset_index(drop=True)
    pool["pair_id"] = np.arange(len(pool), dtype=np.int64)

    uids_tr = set(tr.uid1) | set(tr.uid2)
    uids_te = set(te.uid1) | set(te.uid2)
    print(f"uid overlap between the two files: {len(uids_tr & uids_te)} "
          f"(train {len(uids_tr)}, test {len(uids_te)})")
    print(f"\nshadow pool: {len(pool)} pairs, "
          f"{int((pool.label == 1).sum())} matching / {int((pool.label == 0).sum())} non-matching")

    out = os.path.join(args.out_dir, "shadow_pool.csv")
    pool.to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
