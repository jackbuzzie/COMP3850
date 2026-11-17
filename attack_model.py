"""
Step 2: Membership-inference ATTACK model.

Built on top of train_shadow_models.py - keep that file in the SAME folder. This script
imports its model builders and ClinicalBERT embedder, so the target RLM is built with
exactly the same architecture and settings as the shadow models.

What this script does
---------------------
1. Finds the shadow feature tables made by train_shadow_models.py (any file called
   shadow_attack_features*.npy with a matching shadow_attack_labels*.npy) and works out
   K for each one from the label layout - no renaming or configuring needed.
2. Trains an attack classifier on each table (shadow data ONLY) and checks it on
   shadow models it never saw during training.
3. Builds the "answer key" from the real target:
       target_train.csv rows = true members (1)
       target_test.csv  rows = true non-members (0)
   by training the target RLM (Variant A, same recipe as the shadow models) and
   extracting the same four features from it.
4. Scores the attack against that answer key, saves figures to attack_figures/ and
   a machine-readable attack_results.json. Run show_results.py for a readable report.

Caches (delete a file, or flip a FORCE_* flag, to rebuild it)
-------------------------------------------------------------
  attack_cache/target_emb_*.npy        ClinicalBERT embeddings of the target CSVs
  attack_cache/target_features.npy     features from the target RLM (+ _labels, _meta)
  attack_cache/attack_model_k*.joblib  trained attack models
With everything cached, a re-run takes about a minute and never loads TensorFlow.

ASSUMPTION: the target RLM is Variant A trained on target_train.csv. If the client gave
you a pre-trained target model, replace train_target_and_extract().
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import glob
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, roc_curve

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
SHADOW_GLOB = "shadow_attack_features*.npy"   # labels file = same name with features -> labels
TARGET_TRAIN_PATH = "target_train.csv"
TARGET_TEST_PATH = "target_test.csv"

CACHE_DIR = "attack_cache"
FIG_DIR = "attack_figures"
RESULTS_PATH = "attack_results.json"

FORCE_RECOMPUTE_TARGET = False   # True = retrain the target RLM + re-extract features
FORCE_RETRAIN_ATTACK = False     # True = retrain the attack models
BOOTSTRAP_ROUNDS = 200           # for the AUC confidence interval
SEED = 42

# Must match train_shadow_models.py (Variant A)
FEATURES = ["prob", "dist_from_boundary", "recon_error", "latent_diff_norm"]


def cache_path(name):
    return os.path.join(CACHE_DIR, name)


# ===============================================================
# PART A - find the shadow tables
# ===============================================================
def infer_k(y):
    """
    Each shadow model is stored as [its members][its non-members], one after another.
    Find the block size from that layout and return (K, rows_per_shadow_model).
    """
    n_in = int(np.argmax(y == 0))                       # length of the first run of 1s
    if n_in == 0:
        return None
    later_members = np.flatnonzero(y[n_in:] == 1)       # first 1 after the zeros = start of block 2
    n_per = n_in + int(later_members[0]) if len(later_members) else len(y)
    return (len(y) // n_per, n_per) if len(y) % n_per == 0 else None


def discover_shadow_tables():
    found = {}
    for fp in sorted(glob.glob(SHADOW_GLOB)):
        lp = fp.replace("features", "labels")
        if not os.path.exists(lp):
            print(f"[skip] {fp}: no matching {lp}", flush=True)
            continue
        info = infer_k(np.load(lp).astype(int))
        if info is None:
            print(f"[skip] {fp}: could not work out K from the label layout", flush=True)
            continue
        K = info[0]
        if K in found:
            keep_new = os.path.getmtime(fp) > os.path.getmtime(found[K][0])
            print(f"[note] two files look like K={K}; using the "
                  f"{'newer' if keep_new else 'older'}: {fp if keep_new else found[K][0]}", flush=True)
            if not keep_new:
                continue
        found[K] = (fp, lp)
    return found


# ===============================================================
# PART B - target RLM: embeddings, training, feature extraction
# ===============================================================
def shadow_module():
    """Imports your train_shadow_models.py (loads TensorFlow + ClinicalBERT; only called when needed)."""
    import train_shadow_models as sm
    return sm


def read_pairs(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["text1", "text2", "label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    return df


def get_target_embeddings(tr, te):
    """Embeds the four target text columns with the shadow script's embed_texts, caching each one."""
    parts = {"tr1": tr["text1"], "tr2": tr["text2"], "te1": te["text1"], "te2": te["text2"]}
    paths = {k: cache_path(f"target_emb_{k}.npy") for k in parts}

    def is_valid(k):
        return os.path.exists(paths[k]) and np.load(paths[k], mmap_mode="r").shape[0] == len(parts[k])

    todo = [k for k in parts if not is_valid(k)]
    if todo:
        print("Embedding target text (slow on CPU: ~80,000 texts can take an hour or more)...", flush=True)
        sm = shadow_module()
        for k in todo:
            print(f"  embedding '{k}' ({len(parts[k])} texts)...", flush=True)
            np.save(paths[k], sm.embed_texts(parts[k]))
    else:
        print("Using cached target embeddings.", flush=True)
    return {k: np.load(paths[k]) for k in parts}


def train_target_and_extract(x1_in, x2_in, y_in, x1_out, x2_out, y_out):
    """
    Trains the target RLM on the 'in' data (target_train) and extracts the same four attack
    features as train_shadow_model(), for 'in' rows (members) and 'out' rows (non-members).
    Models come from train_shadow_models.py; the fit settings below (30/20 epochs, batch 256,
    10% validation) must match train_shadow_model() there. It also records training history
    and linkage accuracy, which the shadow script does not keep.
    """
    sm = shadow_module()
    if sm.TARGET_VARIANT != "autoencoder":
        raise SystemExit("This script only supports the 'autoencoder' (Variant A) feature layout.")

    print("Training target RLM stage 1/2: Siamese autoencoder...", flush=True)
    sa_model, encoder = sm.build_siamese_autoencoder()
    sa_model.compile(optimizer="adam", loss=sm.hybrid_loss())
    h_sa = sa_model.fit([x1_in, x2_in], y_in, epochs=30, batch_size=256,
                        validation_split=0.1, verbose=2)

    enc1_in, enc2_in = encoder.predict(x1_in, verbose=0), encoder.predict(x2_in, verbose=0)
    enc1_out, enc2_out = encoder.predict(x1_out, verbose=0), encoder.predict(x2_out, verbose=0)
    diff_in, diff_out = np.abs(enc1_in - enc2_in), np.abs(enc1_out - enc2_out)

    print("Training target RLM stage 2/2: classifier...", flush=True)
    clf = sm.build_classifier(diff_in.shape[1])
    h_clf = clf.fit(diff_in, y_in, epochs=20, batch_size=256, validation_split=0.1, verbose=2)

    prob_in = clf.predict(diff_in, verbose=0).flatten()
    prob_out = clf.predict(diff_out, verbose=0).flatten()

    recon_in = sa_model.predict([x1_in, x2_in], verbose=0)
    recon_out = sa_model.predict([x1_out, x2_out], verbose=0)
    d = recon_in.shape[1] // 2
    err_in = np.mean(np.square(recon_in[:, :d] - recon_in[:, d:]), axis=1)
    err_out = np.mean(np.square(recon_out[:, :d] - recon_out[:, d:]), axis=1)

    feat_in = np.column_stack([prob_in, np.abs(prob_in - 0.5), err_in, np.linalg.norm(diff_in, axis=1)])
    feat_out = np.column_stack([prob_out, np.abs(prob_out - 0.5), err_out, np.linalg.norm(diff_out, axis=1)])

    X = np.vstack([feat_in, feat_out])
    y = np.concatenate([np.ones(len(feat_in)), np.zeros(len(feat_out))])
    meta = {
        "n_members": int(len(feat_in)),
        "n_nonmembers": int(len(feat_out)),
        "linkage_acc_train": float(np.mean((prob_in > 0.5) == (np.asarray(y_in) > 0.5))),
        "linkage_acc_test": float(np.mean((prob_out > 0.5) == (np.asarray(y_out) > 0.5))),
        "history_autoencoder": {k: [float(v) for v in vals] for k, vals in h_sa.history.items()},
        "history_classifier": {k: [float(v) for v in vals] for k, vals in h_clf.history.items()},
    }
    return X, y, meta


def get_target_features():
    f, l, m = (cache_path(n) for n in ("target_features.npy", "target_labels.npy", "target_meta.json"))
    if all(os.path.exists(p) for p in (f, l, m)) and not FORCE_RECOMPUTE_TARGET:
        print("Using cached target features.", flush=True)
        with open(m) as fh:
            return np.load(f), np.load(l), json.load(fh)

    tr, te = read_pairs(TARGET_TRAIN_PATH), read_pairs(TARGET_TEST_PATH)
    print(f"Target data: {len(tr)} train pairs (members), {len(te)} test pairs (non-members)", flush=True)
    emb = get_target_embeddings(tr, te)
    X, y, meta = train_target_and_extract(
        emb["tr1"], emb["tr2"], tr["label"].values.astype(np.float32),
        emb["te1"], emb["te2"], te["label"].values.astype(np.float32))
    np.save(f, X)
    np.save(l, y)
    with open(m, "w") as fh:
        json.dump(meta, fh)
    return X, y, meta


# ===============================================================
# PART C - statistics helpers
# ===============================================================
def cohens_d(a, b):
    """Standardised gap (mean(a) - mean(b)) / pooled std. 0.13 = 'small', 0.5 = 'medium'."""
    pooled = np.sqrt((a.var() + b.var()) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def gap_summary(X, y):
    """Standardised member-minus-non-member gap for every feature."""
    return {f: cohens_d(X[y == 1, j], X[y == 0, j]) for j, f in enumerate(FEATURES)}


def tpr_at_fpr(fpr, tpr, target):
    ok = fpr <= target
    return float(tpr[ok].max()) if ok.any() else 0.0


def bootstrap_auc_ci(y, s, rounds=BOOTSTRAP_ROUNDS, seed=SEED):
    rng = np.random.default_rng(seed)
    n, aucs = len(y), []
    for _ in range(rounds):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        aucs.append(roc_auc_score(y[idx], s[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


# ===============================================================
# PART D - attack model per K
# ===============================================================
def run_attack(K, X, y, X_t, y_t):
    """Train (or load) the attack for one K, validate on unseen shadow models, test on target."""
    n_per = len(y) // K
    shadow_id = np.minimum(np.arange(len(y)) // n_per, K - 1)   # rows are stored one shadow model after another
    n_val = max(1, K // 5)
    val_mask = shadow_id >= K - n_val                            # whole shadow models held out
    sig = f"{X.shape}-{float(X.sum()):.3f}"                      # detects a changed features file

    model_path = cache_path(f"attack_model_k{K}.joblib")
    bundle = joblib.load(model_path) if os.path.exists(model_path) and not FORCE_RETRAIN_ATTACK else None
    if bundle is not None and bundle.get("sig") == sig:
        print(f"  K={K}: loaded cached attack model", flush=True)
        model = bundle["model"]
    else:
        print(f"  K={K}: training attack model on {int((~val_mask).sum())} rows "
              f"({K - n_val} shadow models)...", flush=True)
        model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                               early_stopping=True, random_state=SEED)
        model.fit(X[~val_mask], y[~val_mask])
        joblib.dump({"model": model, "sig": sig}, model_path)

    # 1) validation on shadow models the attack never saw
    Xv, yv = X[val_mask], y[val_mask]
    sv = model.predict_proba(Xv)[:, 1]
    shadow_auc = float(roc_auc_score(yv, sv))
    sfpr, stpr, sthr = roc_curve(yv, sv)
    threshold = float(sthr[np.argmax(stpr - sfpr)])              # Youden's J, chosen on SHADOW data only

    # 2) the real test: target answer key
    st = model.predict_proba(X_t)[:, 1]
    fpr, tpr, _ = roc_curve(y_t, st)
    auc = float(roc_auc_score(y_t, st))
    lo, hi = bootstrap_auc_ci(y_t, st)
    pred = st >= threshold
    tpr_thr = float(pred[y_t == 1].mean())
    fpr_thr = float(pred[y_t == 0].mean())

    # 3) which features matter (permutation importance on held-out shadow rows)
    rng = np.random.default_rng(SEED)
    sub = rng.choice(len(yv), size=min(50000, len(yv)), replace=False)
    imp = permutation_importance(model, Xv[sub], yv[sub], scoring="roc_auc",
                                 n_repeats=3, random_state=SEED)

    summary = {
        "shadow_models_train": K - n_val, "shadow_models_validation": n_val,
        "shadow_val_auc": shadow_auc, "threshold": threshold,
        "target_auc": auc, "target_auc_ci": [lo, hi],
        "tpr_at_1pct_fpr": tpr_at_fpr(fpr, tpr, 0.01),
        "tpr_at_0.1pct_fpr": tpr_at_fpr(fpr, tpr, 0.001),
        "tpr_at_threshold": tpr_thr, "fpr_at_threshold": fpr_thr,
        "balanced_accuracy": (tpr_thr + (1 - fpr_thr)) / 2,
        "single_feature_auc": {f: float(roc_auc_score(y_t, X_t[:, j])) for j, f in enumerate(FEATURES)},
        "permutation_importance": {f: [float(imp.importances_mean[j]), float(imp.importances_std[j])]
                                   for j, f in enumerate(FEATURES)},
    }
    print(f"  K={K}: shadow-validation AUC {shadow_auc:.4f} | TARGET AUC {auc:.4f} "
          f"(95% CI {lo:.4f}-{hi:.4f})", flush=True)
    return summary, (fpr, tpr), st


# ===============================================================
# PART E - figures
# ===============================================================
def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_distribution(X, y, K):
    j = FEATURES.index("dist_from_boundary")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, 0.5, 60)
    for label, name, color in [(0, "Non-members (OUT)", "tab:blue"), (1, "Members (IN)", "tab:orange")]:
        v = X[y == label, j]
        ax.hist(v, bins=bins, alpha=0.55, density=True, label=name, color=color)
        ax.axvline(v.mean(), color=color, linestyle="--", linewidth=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("dist_from_boundary = |confidence - 0.5|   (higher = more decisive)")
    ax.set_ylabel("Density (log scale)")
    ax.set_title(f"Shadow models (K={K}): decisiveness of members vs non-members\n(dashed lines = group means)")
    ax.legend()
    return save(fig, "fig1_shadow_dist_from_boundary.png")


def fig_gaps(shadow_phase, target_gap, per_shadow, ks):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1.6, 1]})
    groups = [(f"Shadow K={k}", shadow_phase[str(k)]["standardised_gap"]) for k in ks]
    groups.append(("Target (real)", target_gap))
    w = 0.8 / len(groups)
    xs = np.arange(len(FEATURES))
    for gi, (name, gap) in enumerate(groups):
        ax1.bar(xs + gi * w - 0.4 + w / 2, [gap[f] for f in FEATURES], w, label=name)
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(FEATURES, rotation=15)
    ax1.set_ylabel("Standardised gap (Cohen's d)\nmember minus non-member")
    ax1.set_title("Size of the member/non-member gap, per feature")
    ax1.legend(fontsize=8)

    rng = np.random.default_rng(SEED)
    for i, k in enumerate(ks):
        d = per_shadow[k]
        ax2.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d, alpha=0.7, color="tab:orange")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.axhline(target_gap["dist_from_boundary"], color="tab:red", linestyle="--", label="Target (real)")
    ax2.set_xticks(range(len(ks)))
    ax2.set_xticklabels([f"K={k}" for k in ks])
    ax2.set_ylabel("Gap (Cohen's d)")
    ax2.set_title("dist_from_boundary: one dot per shadow model")
    ax2.legend(fontsize=8)
    return save(fig, "fig2_effect_size_by_K.png")


def fig_training_curves(meta):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in [(ax1, "history_autoencoder", "Stage 1: Siamese autoencoder"),
                           (ax2, "history_classifier", "Stage 2: classifier")]:
        h = meta.get(key, {})
        for series, label in [("loss", "training loss"), ("val_loss", "validation loss")]:
            if series in h:
                ax.plot(range(1, len(h[series]) + 1), h[series], label=label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend()
    fig.suptitle(f"Target RLM training  (linkage accuracy: train {meta['linkage_acc_train']:.1%}, "
                 f"test {meta['linkage_acc_test']:.1%})")
    return save(fig, "fig3_target_training_curves.png")


def fig_roc(curves):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, log in [(ax1, False), (ax2, True)]:
        for k, (fpr, tpr) in curves.items():
            ax.plot(fpr, tpr, label=f"K={k}")
        ax.plot([1e-4 if log else 0, 1], [1e-4 if log else 0, 1], "k--", label="Random guessing")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        if log:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(1e-4, 1)
            ax.set_ylim(1e-4, 1)
            ax.set_title("Same curves, log scale (low-FPR region)")
        else:
            ax.set_title("Attack ROC on the real target")
        ax.legend()
    return save(fig, "fig4_roc_curves.png")


def fig_scores(scores, y_t, K):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, 1, 50)
    ax.hist(scores[y_t == 0], bins=bins, alpha=0.55, density=True, label="True non-members (target_test)", color="tab:blue")
    ax.hist(scores[y_t == 1], bins=bins, alpha=0.55, density=True, label="True members (target_train)", color="tab:orange")
    ax.set_xlabel("Attack model's predicted probability of 'member'")
    ax.set_ylabel("Density")
    ax.set_title(f"Attack scores on the real target (K={K})")
    ax.legend()
    return save(fig, "fig5_attack_score_hist.png")


def fig_importance(imp, K):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    means = [imp[f][0] for f in FEATURES]
    stds = [imp[f][1] for f in FEATURES]
    ax.barh(FEATURES, means, xerr=stds, color="tab:green")
    ax.set_xlabel("AUC lost when the feature is shuffled")
    ax.set_title(f"Which features does the attack use? (K={K})")
    return save(fig, "fig6_feature_importance.png")


# ===============================================================
# MAIN
# ===============================================================
def main():
    t0 = time.time()
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    found = discover_shadow_tables()
    if not found:
        raise SystemExit(f"No shadow feature files found matching '{SHADOW_GLOB}' in this folder.")
    print("Shadow tables found: " + ", ".join(f"K={k} ({found[k][0]})" for k in sorted(found)), flush=True)

    X_t, y_t, target_meta = get_target_features()
    X_t = X_t.astype(np.float32)
    y_t = y_t.astype(int)
    print(f"Target evaluation set: {int(y_t.sum())} members / {int((y_t == 0).sum())} non-members", flush=True)

    tables = {}
    for K, (fp, lp) in found.items():
        X, y = np.load(fp).astype(np.float32), np.load(lp).astype(int)
        assert X.shape[1] == len(FEATURES), f"{fp} has {X.shape[1]} columns, expected {len(FEATURES)}"
        tables[K] = (X, y)
    ks = sorted(tables)
    k_main = max(ks)

    results = {
        "meta": {"shadow_ks": ks, "main_k": k_main, "features": FEATURES,
                 "shadow_files": {str(k): found[k][0] for k in ks},
                 "target_members": int(y_t.sum()), "target_nonmembers": int((y_t == 0).sum()),
                 "bootstrap_rounds": BOOTSTRAP_ROUNDS},
        "target_rlm": {k: target_meta[k] for k in ("linkage_acc_train", "linkage_acc_test")},
        "target_standardised_gap": gap_summary(X_t, y_t),
        "shadow_phase": {}, "attack_phase": {},
    }

    per_shadow, curves, main_scores = {}, {}, None
    print("\nTraining / evaluating attack models:", flush=True)
    for K in ks:
        X, y = tables[K]
        n_per = len(y) // K
        j = FEATURES.index("dist_from_boundary")
        blocks = [(X[b * n_per:(b + 1) * n_per], y[b * n_per:(b + 1) * n_per]) for b in range(K)]
        per_shadow[K] = [cohens_d(Xb[yb == 1, j], Xb[yb == 0, j]) for Xb, yb in blocks]
        results["shadow_phase"][str(K)] = {
            "rows": int(len(y)),
            "mean_gap": {f: float(X[y == 1, i].mean() - X[y == 0, i].mean()) for i, f in enumerate(FEATURES)},
            "standardised_gap": gap_summary(X, y),
            "per_shadow_model_d_dist_from_boundary": per_shadow[K],
        }
        summary, curve, scores = run_attack(K, X, y, X_t, y_t)
        results["attack_phase"][str(K)] = summary
        curves[K] = curve
        if K == k_main:
            main_scores = scores

    print("\nMaking figures...", flush=True)
    Xm, ym = tables[k_main]
    figs = {
        "shadow_distribution": fig_distribution(Xm, ym, k_main),
        "effect_sizes": fig_gaps(results["shadow_phase"], results["target_standardised_gap"], per_shadow, ks),
        "target_training": fig_training_curves(target_meta),
        "roc": fig_roc(curves),
        "score_hist": fig_scores(main_scores, y_t, k_main),
        "importance": fig_importance(results["attack_phase"][str(k_main)]["permutation_importance"], k_main),
    }
    results["figures"] = figs

    with open(RESULTS_PATH, "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"\nDone in {time.time() - t0:.0f}s. Results -> {RESULTS_PATH}, figures -> {FIG_DIR}/")
    print("Run show_results.py for a readable report.")


if __name__ == "__main__":
    main()
