# Shadow-Model Membership Inference Attack on a Record-Linkage Model

COMP3850, Group 55, Deliverable 7.

**Authors:** Rohini Kamath, Elijah Yang, Mohammed Ali, Jack Buzacott, Deklan McLennan.

Grey-box **membership inference attack (MIA)** against a clinical record-linkage model (RLM). Given a pair of records, the attack predicts whether that pair was in the RLM's training set.

The attack is grey-box: it does not access the target's trained weights, but it uses knowledge of the target architecture and two features derived from internal representations (`recon_error`, `latent_diff_norm`). A strict black-box attacker would only see `prob` and `dist_from_boundary`; that setting was not evaluated.

Threat model: a semi-honest ("honest-but-curious") insider with legitimate query access to the RLM, supplied by the sponsor (MQSOC). The attacker has no access to the target's training data or weights.

Full methodology, test cases and recommendations are in the project report (`COMP3850 D7.pdf`). Code: https://github.com/jackbuzzie/COMP3850

## How it works

The target RLM (Variant A) is:

1. Text pairs (`text1`, `text2`) → frozen [Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) `[CLS]` embeddings (768-d).
2. Siamese autoencoder (768 → 50 latent → 768) trained with a hybrid reconstruction + contrastive loss.
3. Classifier on `|enc1 − enc2|` → match probability.

The attack:

1. **Shadow phase** (`train_shadow_models.py`): pool the shadow data, randomly split 50/50 into in/out, K times. Train a shadow RLM on each "in" half. Record four features per pair, labelled member (1) / non-member (0).
2. **Attack phase** (`attack_model.py`): train a gradient-boosting classifier on the shadow features (last ~20% of shadow models held out for validation). Score it on the real target: `target_train.csv` = members, `target_test.csv` = non-members.

Attack features:

| Feature | Meaning |
|---|---|
| `prob` | Classifier match probability |
| `dist_from_boundary` | `abs(prob − 0.5)`, i.e. confidence |
| `recon_error` | Autoencoder reconstruction error between the pair |
| `latent_diff_norm` | L2 norm of `abs(enc1 − enc2)` |

## Files

| File | Purpose |
|---|---|
| `train_shadow_models.py` | Step 1. Trains K shadow models, writes `shadow_attack_features_k{K}.npy` / `shadow_attack_labels_k{K}.npy`. |
| `attack_model.py` | Step 2. Trains attack models, evaluates on the target, writes `attack_results.json` and `attack_figures/`. |
| `show_results.py` | Step 3. Turns `attack_results.json` into a console summary and `attack_report.html`. Standard library only. |
| `output.py` | Quick inspection of a shadow feature table. Reads unsuffixed `shadow_attack_features.npy` / `shadow_attack_labels.npy`, so rename a `_k{K}` pair or edit the paths first. |
| `shadow_attack_*_k{5,10,15}.npy` | Precomputed shadow feature tables (K = 5, 10, 15). |
| `attack_results.json`, `attack_report.html` | Precomputed results and report. |

## Setup

Python 3.10–3.12 (the report notes TensorFlow has no Python 3.13 support).

```bash
pip install numpy pandas scikit-learn joblib matplotlib tensorflow transformers
```

Versions are not pinned in the repo. `train_shadow_models.py` uses `TFBertModel`, so your `transformers` version must support TensorFlow (newer releases may also need `tf-keras`).

## Data (not included)

Supplied by the sponsor (MQSOC) for this project and not redistributed here. Place these CSVs in the project folder. Each needs columns `text1`, `text2`, `label` (1 = same entity, 0 = different).

The raw `shadow_test.csv` was malformed (238 empty columns, a label column mixing strings and integers, rows with text spilling into the label field). It was cleaned to `shadow_test_clean.csv` (8,001 → 7,999 rows); the cleaning script is not in this repo. Pooled with `shadow_train.csv` this gives 39,999 pairs.

| File | Used for |
|---|---|
| `shadow_train.csv`, `shadow_test_clean.csv` | Shadow pool (paths set in `train_shadow_models.py`) |
| `target_train.csv` | Target members |
| `target_test.csv` | Target non-members |

## Usage

Run from the project folder.

```bash
# 1. Shadow models (slow; embedding the ~55k unique records took ~12 hours in the report's runs)
python train_shadow_models.py

# 2. Attack + evaluation
python attack_model.py

# 3. Readable report
python show_results.py        # writes attack_report.html
```

Skip step 1 to reuse the precomputed `shadow_attack_*_k{5,10,15}.npy`. `attack_model.py` auto-discovers every `shadow_attack_features*.npy` with a matching labels file and infers K from the label layout. Step 2 still needs the target CSVs.

Configuration is at the top of each script: `K`, `SHADOW_TRAIN_FRAC`, `MARGIN`, `ALPHA`, `RANDOM_SEED` (42) in `train_shadow_models.py`; `FORCE_RECOMPUTE_TARGET`, `FORCE_RETRAIN_ATTACK`, `BOOTSTRAP_ROUNDS` in `attack_model.py`.

### Caches

Delete a file or set the matching `FORCE_*` flag to rebuild.

| Path | Contents |
|---|---|
| `x1_pool.npy`, `x2_pool.npy` | Shadow-pool embeddings |
| `attack_cache/target_emb_*.npy` | Target embeddings |
| `attack_cache/target_features.npy` (+ labels, meta) | Target RLM features |
| `attack_cache/attack_model_k*.joblib` | Trained attack models |

With everything cached, step 2 takes about a minute and does not load TensorFlow.

## Results (included run)

Target: 32,000 members / 8,000 non-members. Target RLM linkage accuracy: 96.0% train, 93.4% test.

| K | Shadow-val AUC | Target AUC (95% CI) | TPR @ 1% FPR |
|---|---|---|---|
| 5 | 0.554 | 0.549 (0.543–0.556) | 1.6% |
| 10 | 0.556 | 0.549 (0.542–0.557) | 1.7% |
| 15 | 0.560 | 0.547 (0.541–0.554) | 1.5% |

Conclusion: a **very weak leak**. It is statistically detectable (CI excludes 0.5) but barely better than random guessing. `dist_from_boundary` carries the clearest member/non-member gap (Cohen's d ≈ 0.13 on shadow, 0.09 on target); `prob` and `latent_diff_norm` show almost none.

## Recommendations (from the report)

- Limit exposed confidence scores and internal representations (e.g. return match/non-match or rounded probabilities).
- Monitor the train/test accuracy gap across retraining; re-run this pipeline after every retrain.
- Consider DP-SGD for stronger guarantees (untested; may cost linkage accuracy).
- Apply least-privilege access, query logging and audits.

These results apply only to the tested attack and are not a guarantee against stronger methods.

## Limitations

- The target RLM is retrained from `target_train.csv` using the Variant A recipe. A client-supplied pre-trained model may behave differently; replace `train_target_and_extract()` to attack it.
- Shadow models train on ~half the shadow pool (~20k pairs) vs. 32k for the target, so overfitting levels differ.
- Membership is per record pair; some record IDs appear in both target train and test pairs.
- The target set is ~80% members, so prefer AUC and TPR at low FPR over raw accuracy.
- The K = 5, 10 and 15 runs share seed 42, so their splits overlap; treat them as variations of one method, not independent experiments.
- 10% of each model's member data is held out as a validation split, so not every member pair updates the weights.
- Only Variant A is attacked. `train_shadow_models.py` also contains a `raw_diff` Variant B, but `attack_model.py` rejects it.
- Results apply to the grey-box setting only, not a strict black-box attacker.
- The member/non-member answer key exists only in this simulated setup; a real attacker would not have it.
