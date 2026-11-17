"""
Step 1: Train K shadow models and save their pooled attack-feature table.

This does NOT train the attack model yet, and does NOT touch the real
target's data - it only produces the labeled (feature -> member/non-member)
table that the attack model will later be trained on.

Edit SHADOW_TRAIN_PATH / SHADOW_TEST_PATH below to point at your two shadow
files. They get pooled together, then re-split K different times - the
original shadow_train/shadow_test boundary is NOT reused as one of the K
splits.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from transformers import BertTokenizer, TFBertModel

# ---------------------------------------------------------------
# Config - edit these paths to match your files
# ---------------------------------------------------------------
SHADOW_TRAIN_PATH = "shadow_train.csv"
SHADOW_TEST_PATH = "shadow_test_clean.csv"   # use the cleaned version

K = 15
SHADOW_TRAIN_FRAC = 0.5
EMBEDDING_DIM = 768
LATENT_DIM = 50
MARGIN = 2.5
ALPHA = 1.0
RANDOM_SEED = 42
TARGET_VARIANT = "autoencoder"   # "autoencoder" (Variant A) or "raw_diff" (Variant B)

OUTPUT_FEATURES_PATH = f"shadow_attack_features_k{K}.npy"
OUTPUT_LABELS_PATH = f"shadow_attack_labels_k{K}.npy"

rng = np.random.default_rng(RANDOM_SEED)

# ---------------------------------------------------------------
# Step 0: Frozen embedding extractor - computed ONCE for the whole pool
# and reused across all K shadow models (it never changes between them)
# ---------------------------------------------------------------
print("Loading ClinicalBERT (downloads weights on first run, cached after)...", flush=True)
tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
bert_model = TFBertModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
print("ClinicalBERT loaded.", flush=True)


def embed_texts(texts, batch_size=32, max_length=512):
    embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    for batch_num, i in enumerate(range(0, len(texts), batch_size), start=1):
        batch = texts.iloc[i:i + batch_size].tolist()
        inputs = tokenizer(
            batch, return_tensors="tf", padding=True,
            truncation=True, max_length=max_length
        )
        outputs = bert_model(inputs["input_ids"])
        embeddings.append(outputs.last_hidden_state[:, 0, :].numpy())
        if batch_num % 20 == 0 or batch_num == total_batches:
            print(f"  batch {batch_num}/{total_batches}", flush=True)
    return np.vstack(embeddings)


# ---------------------------------------------------------------
# Step 1: Model builders - mirror the target architecture exactly
# ---------------------------------------------------------------
def build_siamese_autoencoder(embedding_dim=EMBEDDING_DIM, latent_dim=LATENT_DIM):
    encoder_input = Input(shape=(embedding_dim,))
    x = layers.Dense(latent_dim, activity_regularizer=regularizers.l1(0.01))(encoder_input)
    x = layers.LeakyReLU(alpha=0.01)(x)
    encoder_output = layers.Dense(embedding_dim, activation="relu")(x)
    encoder = Model(encoder_input, encoder_output, name="encoder")

    decoder_input = Input(shape=(embedding_dim,))
    decoder_output = layers.Dense(embedding_dim, activation="sigmoid")(decoder_input)
    decoder = Model(decoder_input, decoder_output, name="decoder")

    in1, in2 = Input(shape=(embedding_dim,)), Input(shape=(embedding_dim,))
    enc1, enc2 = encoder(in1), encoder(in2)
    rec1, rec2 = decoder(enc1), decoder(enc2)
    merged = layers.Concatenate()([rec1, rec2])
    sa_model = Model(inputs=[in1, in2], outputs=merged, name="siamese_autoencoder")
    return sa_model, encoder


def hybrid_loss(margin=MARGIN, alpha=ALPHA):
    def loss_fn(y_true, y_pred):
        emb_dim = tf.shape(y_pred)[1] // 2
        recon1, recon2 = y_pred[:, :emb_dim], y_pred[:, emb_dim:]
        recon_loss = tf.reduce_mean(tf.square(recon1 - recon2), axis=1)
        dist = tf.sqrt(tf.reduce_sum(tf.square(recon1 - recon2), axis=1) + 1e-8)
        y_true = tf.cast(y_true, tf.float32)
        contrastive = y_true * tf.square(dist) + (1 - y_true) * tf.square(tf.maximum(margin - dist, 0))
        return tf.reduce_mean(alpha * recon_loss + contrastive)
    return loss_fn


def build_classifier(input_dim, dropout=0.5, l2=0.01):
    inp = Input(shape=(input_dim,))
    x = Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2))(inp)
    x = Dropout(dropout)(x)
    x = Dense(32, activation="relu", kernel_regularizer=regularizers.l2(l2))(x)
    out = Dense(1, activation="sigmoid")(x)
    clf = Model(inp, out)
    clf.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return clf


# ---------------------------------------------------------------
# Step 2: Train one shadow model, extract attack features for its
# in (member) and out (non-member) rows
# ---------------------------------------------------------------
def train_shadow_model(x1_in, x2_in, y_in, x1_out, x2_out, y_out, variant=TARGET_VARIANT):
    if variant == "autoencoder":
        sa_model, encoder = build_siamese_autoencoder()
        sa_model.compile(optimizer="adam", loss=hybrid_loss())
        sa_model.fit([x1_in, x2_in], y_in, epochs=30, batch_size=256,
                     validation_split=0.1, verbose=0)

        enc1_in, enc2_in = encoder.predict(x1_in, verbose=0), encoder.predict(x2_in, verbose=0)
        enc1_out, enc2_out = encoder.predict(x1_out, verbose=0), encoder.predict(x2_out, verbose=0)
        diff_in, diff_out = np.abs(enc1_in - enc2_in), np.abs(enc1_out - enc2_out)

        clf = build_classifier(diff_in.shape[1])
        clf.fit(diff_in, y_in, epochs=20, batch_size=256, validation_split=0.1, verbose=0)

        prob_in = clf.predict(diff_in, verbose=0).flatten()
        prob_out = clf.predict(diff_out, verbose=0).flatten()

        recon_in = sa_model.predict([x1_in, x2_in], verbose=0)
        recon_out = sa_model.predict([x1_out, x2_out], verbose=0)
        d = recon_in.shape[1] // 2
        recon_err_in = np.mean(np.square(recon_in[:, :d] - recon_in[:, d:]), axis=1)
        recon_err_out = np.mean(np.square(recon_out[:, :d] - recon_out[:, d:]), axis=1)

        feat_in = np.column_stack([prob_in, np.abs(prob_in - 0.5), recon_err_in,
                                    np.linalg.norm(diff_in, axis=1)])
        feat_out = np.column_stack([prob_out, np.abs(prob_out - 0.5), recon_err_out,
                                     np.linalg.norm(diff_out, axis=1)])

    else:  # "raw_diff" - Variant B
        diff_in, diff_out = np.abs(x1_in - x2_in), np.abs(x1_out - x2_out)

        clf = build_classifier(diff_in.shape[1])
        clf.fit(diff_in, y_in, epochs=30, batch_size=256, validation_split=0.1, verbose=0)

        prob_in = clf.predict(diff_in, verbose=0).flatten()
        prob_out = clf.predict(diff_out, verbose=0).flatten()

        feat_in = np.column_stack([prob_in, np.abs(prob_in - 0.5), np.linalg.norm(diff_in, axis=1)])
        feat_out = np.column_stack([prob_out, np.abs(prob_out - 0.5), np.linalg.norm(diff_out, axis=1)])

    labels_in = np.ones(len(feat_in))     # member
    labels_out = np.zeros(len(feat_out))  # non-member
    return np.vstack([feat_in, feat_out]), np.concatenate([labels_in, labels_out])


# ---------------------------------------------------------------
# Step 3: Run K shadow models, pool their attack-feature outputs
# ---------------------------------------------------------------
def run_shadow_pipeline(x1_pool, x2_pool, y_pool, k=K, shadow_train_frac=SHADOW_TRAIN_FRAC):
    all_features, all_labels = [], []
    n = len(y_pool)

    for i in range(k):
        idx = rng.permutation(n)
        split = int(n * shadow_train_frac)
        in_idx, out_idx = idx[:split], idx[split:]

        feats, labs = train_shadow_model(
            x1_pool[in_idx], x2_pool[in_idx], y_pool[in_idx],
            x1_pool[out_idx], x2_pool[out_idx], y_pool[out_idx],
        )
        all_features.append(feats)
        all_labels.append(labs)
        print(f"[shadow {i + 1}/{k}] done - {len(in_idx)} in / {len(out_idx)} out")

    return np.vstack(all_features), np.concatenate(all_labels)

if __name__ == "__main__":
    # Pool the two shadow files together - they get re-split K times below,
    # the original shadow_train/shadow_test boundary is not reused directly.
    shadow_train_df = pd.read_csv(SHADOW_TRAIN_PATH)
    shadow_test_df = pd.read_csv(SHADOW_TEST_PATH)
    pool_df = pd.concat([shadow_train_df, shadow_test_df], ignore_index=True)
    print(f"Pooled shadow data: {len(pool_df)} rows", flush=True)

    if os.path.exists("x1_pool.npy") and os.path.exists("x2_pool.npy"):
        print("Found cached embeddings - loading instead of recomputing.", flush=True)
        x1_pool = np.load("x1_pool.npy")
        x2_pool = np.load("x2_pool.npy")
    else:
        print("Embedding text1 with ClinicalBERT (done once, reused across all K models)...", flush=True)
        x1_pool = embed_texts(pool_df["text1"])
        print("Embedding text2 with ClinicalBERT...", flush=True)
        x2_pool = embed_texts(pool_df["text2"])
        np.save("x1_pool.npy", x1_pool)
        np.save("x2_pool.npy", x2_pool)
        print("Embeddings computed and cached to x1_pool.npy / x2_pool.npy.", flush=True)

    y_pool = pool_df["label"].values.astype(np.float32)
    print("Embeddings ready. Starting shadow model training loop...", flush=True)

    X_attack, y_attack = run_shadow_pipeline(x1_pool, x2_pool, y_pool, k=K)

    np.save(OUTPUT_FEATURES_PATH, X_attack)
    np.save(OUTPUT_LABELS_PATH, y_attack)
    print(f"Saved pooled attack-feature table: {X_attack.shape} -> {OUTPUT_FEATURES_PATH}")
    print(f"Saved labels: {y_attack.shape} -> {OUTPUT_LABELS_PATH}")