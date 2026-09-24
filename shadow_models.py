"""
Step 2 - Train k shadow models and build the membership-inference attack dataset.

Each shadow model replicates the target pipeline in SNN_Final_end_to_end.ipynb:
    siamese autoencoder (50-unit bottleneck, LeakyReLU, L1 activity penalty)
      -> hybrid recon + contrastive loss, margin 2.5, alpha 1.0, 30 epochs
      -> MLP head on |enc1 - enc2| (64 L2 + dropout 0.5 -> 32 L2 -> sigmoid)
      -> decision threshold tuned for F1 on a held-out 20% of its own members

Shadow model s trains on a random half of the shadow pool (its MEMBERS) and
never sees the other half (its NON-MEMBERS). Over k models each pair lands
in-training about k/2 times, so the output supports both attack styles:
  * global / Shokri: pool all (features, member?) rows, fit one attack model
  * per-example / LiRA: group rows by pair_id, compare each pair's in- and
    out-distributions (that is why per_model/ keeps the raw rows)

By default the split is uid-disjoint: pairs sharing a record are kept on the
same side. Without this, a "non-member" pair can contain a record the shadow
model already trained on through a different pair, which blunts the membership
signal the attack is supposed to measure. This dataset has 19,211 connected
components over 39,999 pairs, so a near-balanced disjoint split is easy.

Usage:
    python shadow_models.py --k 16
    python shadow_models.py --k 16 --no-uid-disjoint --sa-epochs 10 --clf-epochs 10
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Keras 2 / Keras 3 compatibility
# ---------------------------------------------------------------------------
def _leaky_relu(slope):
    from tensorflow.keras import layers
    try:
        return layers.LeakyReLU(negative_slope=slope)   # Keras 3
    except TypeError:
        return layers.LeakyReLU(alpha=slope)            # Keras 2


# ---------------------------------------------------------------------------
# Model definitions - kept identical to the target notebook
# ---------------------------------------------------------------------------
def build_siamese_autoencoder(embedding_dim):
    from tensorflow.keras import layers, regularizers
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input

    encoder_input = Input(shape=(embedding_dim,))
    x = layers.Dense(50, activity_regularizer=regularizers.l1(0.01))(encoder_input)
    x = _leaky_relu(0.01)(x)
    encoder_output = layers.Dense(embedding_dim, activation="relu")(x)
    encoder = Model(encoder_input, encoder_output)

    decoder_input = Input(shape=(embedding_dim,))
    decoder_output = layers.Dense(embedding_dim, activation="sigmoid")(decoder_input)
    decoder = Model(decoder_input, decoder_output)

    input1, input2 = Input(shape=(embedding_dim,)), Input(shape=(embedding_dim,))
    merged = layers.Concatenate()([decoder(encoder(input1)), decoder(encoder(input2))])
    return Model(inputs=[input1, input2], outputs=merged), encoder


def hybrid_classification_loss(margin=2.5, alpha=1.0):
    import tensorflow as tf

    def loss_fn(y_true, y_pred):
        emb_dim = tf.shape(y_pred)[1] // 2
        recon1, recon2 = y_pred[:, :emb_dim], y_pred[:, emb_dim:]
        recon_loss = tf.reduce_mean(tf.square(recon1 - recon2), axis=1)
        distances = tf.sqrt(tf.reduce_sum(tf.square(recon1 - recon2), axis=1) + 1e-12)
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        contrastive = (y_true * tf.square(distances)
                       + (1 - y_true) * tf.square(tf.maximum(margin - distances, 0)))
        return tf.reduce_mean(alpha * recon_loss + contrastive)

    return loss_fn


def build_classifier(input_dim):
    from tensorflow.keras import regularizers
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, Dropout

    inp = Input(shape=(input_dim,))
    x = Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.01))(inp)
    x = Dropout(0.5)(x)
    x = Dense(32, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
    out = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def component_labels(pool):
    """Connected components over the uid graph; pairs sharing a record group."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    uids = pd.Index(pd.unique(pd.concat([pool.uid1, pool.uid2])))
    i, j = uids.get_indexer(pool.uid1), uids.get_indexer(pool.uid2)
    adj = coo_matrix((np.ones(len(pool)), (i, j)), shape=(len(uids), len(uids)))
    _, lab = connected_components(adj, directed=False)
    return lab[i]


def member_mask(n_pairs, comps, rng, frac=0.5):
    """Boolean mask of members, balanced to ~frac of pairs."""
    if comps is None:
        idx = rng.permutation(n_pairs)
        mask = np.zeros(n_pairs, dtype=bool)
        mask[idx[:int(round(frac * n_pairs))]] = True
        return mask

    order = rng.permutation(np.unique(comps))
    sizes = pd.Series(comps).value_counts()
    target, running, chosen = frac * n_pairs, 0, set()
    for c in order:
        if running >= target:
            break
        chosen.add(c)
        running += sizes[c]
    return np.isin(comps, list(chosen))


# ---------------------------------------------------------------------------
# Attack features
# ---------------------------------------------------------------------------
def stable_logit(p, y, eps=1e-6):
    """Carlini's stable logit: confidence in the TRUE class, logit-scaled."""
    p = np.clip(p, eps, 1 - eps)
    p_true = np.where(y == 1, p, 1 - p)
    p_true = np.clip(p_true, eps, 1 - eps)
    return np.log(p_true / (1 - p_true))


def extract_features(prob, y, enc1, enc2, threshold):
    eps = 1e-6
    p = np.clip(prob, eps, 1 - eps)
    bce = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    dist = np.sqrt(np.sum((enc1 - enc2) ** 2, axis=1))
    recon = np.mean((enc1 - enc2) ** 2, axis=1)
    return {
        "prob": prob.astype(np.float32),
        "conf_true": np.where(y == 1, prob, 1 - prob).astype(np.float32),
        "loss": bce.astype(np.float32),
        "stable_logit": stable_logit(prob, y).astype(np.float32),
        "correct": ((prob >= threshold).astype(int) == y).astype(np.float32),
        "enc_distance": dist.astype(np.float32),
        "enc_recon_mse": recon.astype(np.float32),
        "task_label": y.astype(np.float32),
    }


FEATURE_ORDER = ["prob", "conf_true", "loss", "stable_logit",
                 "correct", "enc_distance", "enc_recon_mse", "task_label"]


# ---------------------------------------------------------------------------
# One shadow model
# ---------------------------------------------------------------------------
def train_one(s, x1, x2, y, comps, args, out_dir):
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score

    seed = args.seed + s
    rng = np.random.default_rng(seed)
    tf.keras.utils.set_random_seed(seed)

    n = len(y)
    is_member = member_mask(n, comps, rng, args.member_frac)
    mi = np.where(is_member)[0]
    print(f"[shadow {s:02d}] members {len(mi)} / non-members {n - len(mi)}", flush=True)

    # --- siamese autoencoder on this model's members only ---
    sa, encoder = build_siamese_autoencoder(x1.shape[1])
    sa.compile(optimizer="adam",
               loss=hybrid_classification_loss(args.margin, args.alpha))
    sa.fit([x1[mi], x2[mi]], y[mi], epochs=args.sa_epochs,
           batch_size=args.batch_size, validation_split=0.1, verbose=args.verbose)

    # --- classification head, with 20% of members held out for threshold tuning ---
    enc1_all = encoder.predict(x1, batch_size=1024, verbose=0)
    enc2_all = encoder.predict(x2, batch_size=1024, verbose=0)
    diff_all = np.abs(enc1_all - enc2_all)

    tr_idx, val_idx = train_test_split(mi, test_size=0.2, random_state=seed,
                                       stratify=y[mi])
    clf = build_classifier(diff_all.shape[1])
    clf.fit(diff_all[tr_idx], y[tr_idx], epochs=args.clf_epochs,
            batch_size=args.batch_size, validation_split=0.1, verbose=args.verbose)

    val_prob = clf.predict(diff_all[val_idx], batch_size=1024, verbose=0).flatten()
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.1, 0.9, 0.01):
        _, _, f1, _ = precision_recall_fscore_support(
            y[val_idx], (val_prob >= t).astype(int), average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)

    # --- score EVERY pool pair: members and non-members alike ---
    prob_all = clf.predict(diff_all, batch_size=1024, verbose=0).flatten()
    feats = extract_features(prob_all, y, enc1_all, enc2_all, best_t)

    pred = (prob_all >= best_t).astype(int)
    util = {
        "shadow_id": s,
        "seed": seed,
        "n_members": int(len(mi)),
        "n_nonmembers": int(n - len(mi)),
        "best_threshold": best_t,
        "val_f1_at_threshold": float(best_f1),
        # The member/non-member accuracy GAP is the generalisation gap the
        # attack exploits - if it is ~0 the attack has little to find.
        "member_acc": float(accuracy_score(y[is_member], pred[is_member])),
        "nonmember_acc": float(accuracy_score(y[~is_member], pred[~is_member])),
        "member_f1": float(f1_score(y[is_member], pred[is_member], zero_division=0)),
        "nonmember_f1": float(f1_score(y[~is_member], pred[~is_member], zero_division=0)),
        "mean_loss_member": float(feats["loss"][is_member].mean()),
        "mean_loss_nonmember": float(feats["loss"][~is_member].mean()),
    }
    util["acc_gap"] = util["member_acc"] - util["nonmember_acc"]
    print(f"[shadow {s:02d}] thr {best_t:.2f} | member acc {util['member_acc']:.4f} "
          f"| non-member acc {util['nonmember_acc']:.4f} | gap {util['acc_gap']:+.4f}",
          flush=True)

    np.savez_compressed(
        os.path.join(out_dir, "per_model", f"shadow_{s:03d}.npz"),
        pair_id=np.arange(n, dtype=np.int64),
        member=is_member.astype(np.int8),
        **feats)

    if args.save_weights:
        wdir = os.path.join(out_dir, "weights", f"shadow_{s:03d}")
        os.makedirs(wdir, exist_ok=True)
        encoder.save(os.path.join(wdir, "encoder.keras"))
        clf.save(os.path.join(wdir, "classifier.keras"))

    tf.keras.backend.clear_session()
    return util


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=16, help="number of shadow models")
    ap.add_argument("--pool", default="shadow_clean/shadow_pool.csv")
    ap.add_argument("--emb-dir", default="shadow_emb")
    ap.add_argument("--out-dir", default="shadow_out")
    ap.add_argument("--member-frac", type=float, default=0.5)
    ap.add_argument("--uid-disjoint", dest="uid_disjoint", action="store_true", default=True)
    ap.add_argument("--no-uid-disjoint", dest="uid_disjoint", action="store_false")
    ap.add_argument("--sa-epochs", type=int, default=30)
    ap.add_argument("--clf-epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--margin", type=float, default=2.5)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--save-weights", action="store_true")
    ap.add_argument("--verbose", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out_dir, "per_model"), exist_ok=True)

    pool = pd.read_csv(args.pool)
    emb = np.load(os.path.join(args.emb_dir, "uid_embeddings.npy"))
    uid_index = pd.Index(np.load(os.path.join(args.emb_dir, "uid_index.npy"),
                                 allow_pickle=True).astype(str))

    i1 = uid_index.get_indexer(pool.uid1.astype(str))
    i2 = uid_index.get_indexer(pool.uid2.astype(str))
    if (i1 < 0).any() or (i2 < 0).any():
        raise ValueError("some pool uids are missing from the embedding cache - "
                         "re-run embed_shadow.py against this pool file")
    x1, x2, y = emb[i1], emb[i2], pool.label.to_numpy().astype(np.float32)

    # Guard against the x1/x2 mix-up that is present in the target notebook's
    # commented-out save block (x2_test.npy was written from x1_test).
    if np.allclose(x1, x2):
        raise ValueError("x1 and x2 are identical - the embedding cache is wrong")
    print(f"pool {len(pool)} pairs | embedding dim {x1.shape[1]} | "
          f"mean |x1-x2| {np.mean(np.abs(x1 - x2)):.5f}")

    comps = component_labels(pool) if args.uid_disjoint else None
    print(f"split mode: {'uid-disjoint' if args.uid_disjoint else 'random pair-level'}")

    t0 = time.time()
    utils = [train_one(s, x1, x2, y, comps, args, args.out_dir) for s in range(args.k)]

    # --- assemble the attack dataset ---
    rows, mems, ids, models = [], [], [], []
    for s in range(args.k):
        d = np.load(os.path.join(args.out_dir, "per_model", f"shadow_{s:03d}.npz"))
        rows.append(np.column_stack([d[f] for f in FEATURE_ORDER]))
        mems.append(d["member"])
        ids.append(d["pair_id"])
        models.append(np.full(len(d["member"]), s, dtype=np.int16))

    X = np.concatenate(rows).astype(np.float32)
    m = np.concatenate(mems).astype(np.int8)
    np.savez_compressed(os.path.join(args.out_dir, "attack_dataset.npz"),
                        X=X, membership=m,
                        pair_id=np.concatenate(ids),
                        shadow_id=np.concatenate(models),
                        feature_names=np.array(FEATURE_ORDER))

    summary = pd.DataFrame(utils)
    summary.to_csv(os.path.join(args.out_dir, "shadow_summary.csv"), index=False)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\n=== {args.k} shadow models in {time.time() - t0:.0f}s ===")
    print(f"attack dataset X {X.shape}, members {int(m.sum())} / {len(m)}")
    print(f"mean member/non-member accuracy gap: {summary.acc_gap.mean():+.4f} "
          f"(sd {summary.acc_gap.std():.4f})")
    if abs(summary.acc_gap.mean()) < 0.01:
        print("NOTE: the gap is near zero - these shadow models barely overfit, "
              "so expect a weak attack. Raising --sa-epochs/--clf-epochs or "
              "shrinking --member-frac will widen it.")
    print(f"wrote {args.out_dir}/attack_dataset.npz and shadow_summary.csv")


if __name__ == "__main__":
    main()
