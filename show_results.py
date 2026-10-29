"""
Turns attack_results.json (written by attack_model.py) into something readable:
  1. a plain-English summary printed to the console
  2. attack_report.html - open it in a browser; figures are embedded in the file

Needs only the standard library, so it runs instantly and never touches TensorFlow.
Run it as many times as you like - it does not retrain anything.
"""

import base64
import html
import json
import os

RESULTS_PATH = "attack_results.json"
OUT_HTML = "attack_report.html"

FIGURE_NOTES = {
    "shadow_distribution": (
        "Shadow phase: is there any membership signal?",
        "Two overlaid histograms of dist_from_boundary (how decisive the model was). Orange = pairs the "
        "shadow models trained on, blue = pairs they did not. Heavily overlapping shapes with the dashed "
        "mean lines only slightly apart means a weak leak; clearly separated shapes would mean a strong one."),
    "effect_sizes": (
        "Shadow phase: is the gap real, and does it repeat?",
        "Left: the member/non-member gap per feature on a common scale (Cohen's d; roughly 0.2 = small, "
        "0.5 = medium). Bars that keep the same sign and size across K are signal; bars that flip around "
        "are noise. The last bar group is the real target, so you can see whether the shadow gap carries "
        "over. Right: one dot per shadow model - dots mostly above zero mean the effect repeats."),
    "target_training": (
        "Validity check: did the target RLM learn the linkage task?",
        "Training vs validation loss for both stages of the target model. If both curves fall and level off, "
        "the model learned. A growing gap between them is overfitting, which is what membership inference exploits."),
    "roc": (
        "Attack phase: how well does the attack beat guessing?",
        "ROC curves on the real target. The dashed diagonal is random guessing. The further a curve bows "
        "toward the top-left, the stronger the attack. The right panel zooms into the low false-positive "
        "region, which matters most for privacy (a few confident, correct guesses are the real risk)."),
    "score_hist": (
        "Attack phase: do the scores separate members from non-members?",
        "What the attack predicted for the real target's true members (orange) and true non-members (blue). "
        "If the two histograms overlap almost completely, the attack cannot tell them apart."),
    "importance": (
        "Attack phase: which features does the attack lean on?",
        "How much the attack's AUC drops when each feature is shuffled. Longer bar = more important. "
        "Expect dist_from_boundary to dominate if the confidence leak is the main signal."),
}

LIMITATIONS = [
    "The target RLM was retrained from target_train.csv using the notebook's Variant A recipe; if the client "
    "supplied a pre-trained target model, results may differ.",
    "Shadow models were trained on about half the shadow pool (roughly 20,000 pairs), whereas the target was "
    "trained on 32,000 pairs, so the shadow models may not overfit to exactly the same degree as the target.",
    "Membership is judged per record pair. Some individual record IDs appear in both target_train and "
    "target_test (inside different pairs), so a few 'non-member' pairs contain records the model has seen.",
    "The evaluation set is imbalanced (about 80% members), so AUC and TPR at low FPR are more trustworthy "
    "than raw accuracy.",
    "Only the Variant A architecture (autoencoder + classifier) was attacked.",
    "The target answer key is only available in this simulated environment; a real black-box attacker would "
    "not have it, so these numbers are an evaluation, not something an attacker could compute.",
]


def verdict(auc, lo, hi):
    if lo <= 0.5 <= hi:
        return "Not distinguishable from random guessing"
    if hi < 0.5:
        return ("Worse than guessing: the attack points the wrong way on the real target "
                "(shadow models behave differently from the target)")
    if auc < 0.55:
        return "Very weak leak (statistically detectable, but tiny)"
    if auc < 0.65:
        return "Weak-to-moderate leak"
    if auc < 0.75:
        return "Moderate leak"
    return "Strong leak"


def rows_for_table(res):
    rows = []
    for k in res["meta"]["shadow_ks"]:
        a = res["attack_phase"][str(k)]
        lo, hi = a["target_auc_ci"]
        rows.append({
            "K": k,
            "shadow_auc": f"{a['shadow_val_auc']:.3f}",
            "target_auc": f"{a['target_auc']:.3f}",
            "ci": f"{lo:.3f} - {hi:.3f}",
            "tpr1": f"{a['tpr_at_1pct_fpr']:.1%}",
            "tpr01": f"{a['tpr_at_0.1pct_fpr']:.1%}",
            "bacc": f"{a['balanced_accuracy']:.1%}",
            "verdict": verdict(a["target_auc"], lo, hi),
        })
    return rows


def print_console(res):
    rlm = res["target_rlm"]
    m = res["meta"]
    print("=" * 78)
    print("MEMBERSHIP INFERENCE ATTACK - RESULTS")
    print("=" * 78)
    print(f"Evaluation set (real target): {m['target_members']} members / {m['target_nonmembers']} non-members")
    print(f"Target RLM linkage accuracy: train {rlm['linkage_acc_train']:.1%}, test {rlm['linkage_acc_test']:.1%} "
          f"(gap {rlm['linkage_acc_train'] - rlm['linkage_acc_test']:+.1%})")
    print("\nHow to read: AUC 0.5 = random guessing, 1.0 = perfect. TPR@1%FPR for a random guesser is 1%.\n")
    header = f"{'K':>3} | {'Shadow AUC':>10} | {'Target AUC':>10} | {'95% CI':>15} | {'TPR@1%':>7} | {'TPR@0.1%':>8} | {'Bal.acc':>7}"
    print(header)
    print("-" * len(header))
    for r in rows_for_table(res):
        print(f"{r['K']:>3} | {r['shadow_auc']:>10} | {r['target_auc']:>10} | {r['ci']:>15} | "
              f"{r['tpr1']:>7} | {r['tpr01']:>8} | {r['bacc']:>7}   {r['verdict']}")

    k = str(m["main_k"])
    a = res["attack_phase"][k]
    print(f"\nSingle-feature AUC on the target (K={k} run; 0.5 = no signal):")
    for f, v in a["single_feature_auc"].items():
        print(f"  {f:<20} {v:.3f}")
    print("\nStandardised member-minus-non-member gap (Cohen's d): shadow vs real target")
    print(f"  {'feature':<20} {'shadow K=' + k:>12} {'target':>10}")
    for f in m["features"]:
        print(f"  {f:<20} {res['shadow_phase'][k]['standardised_gap'][f]:>12.3f} "
              f"{res['target_standardised_gap'][f]:>10.3f}")
    print("\nLimitations to mention in the report:")
    for item in LIMITATIONS:
        print(f"  - {item}")
    print("=" * 78)


def embed_image(path):
    with open(path, "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode()


def build_html(res):
    rlm, m = res["target_rlm"], res["meta"]
    k = str(m["main_k"])
    rows = "".join(
        f"<tr><td>{r['K']}</td><td>{r['shadow_auc']}</td><td>{r['target_auc']}</td><td>{r['ci']}</td>"
        f"<td>{r['tpr1']}</td><td>{r['tpr01']}</td><td>{r['bacc']}</td><td>{html.escape(r['verdict'])}</td></tr>"
        for r in rows_for_table(res))

    feat_rows = "".join(
        f"<tr><td>{f}</td><td>{res['attack_phase'][k]['single_feature_auc'][f]:.3f}</td>"
        f"<td>{res['shadow_phase'][k]['standardised_gap'][f]:.3f}</td>"
        f"<td>{res['target_standardised_gap'][f]:.3f}</td></tr>" for f in m["features"])

    figs = ""
    for key, path in res.get("figures", {}).items():
        if key in FIGURE_NOTES and os.path.exists(path):
            title, note = FIGURE_NOTES[key]
            figs += (f"<section><h3>{html.escape(title)}</h3><img src='{embed_image(path)}' alt='{key}'>"
                     f"<p class='note'>{html.escape(note)}</p></section>")

    limits = "".join(f"<li>{html.escape(x)}</li>" for x in LIMITATIONS)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>MIA results</title><style>
body{{font-family:system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;line-height:1.5;color:#222}}
table{{border-collapse:collapse;width:100%;margin:1rem 0}}th,td{{border:1px solid #ccc;padding:6px 10px;text-align:left}}
th{{background:#f2f2f2}}img{{max-width:100%}}.note{{color:#555;font-size:.92rem}}section{{margin:2rem 0}}
.box{{background:#f7f7f7;border-left:4px solid #888;padding:.6rem 1rem;margin:1rem 0}}</style></head><body>
<h1>Membership inference attack - results</h1>
<div class="box">Evaluated on the real target: <b>{m['target_members']}</b> true members (target_train) and
<b>{m['target_nonmembers']}</b> true non-members (target_test).<br>
Target RLM linkage accuracy: train <b>{rlm['linkage_acc_train']:.1%}</b>, test <b>{rlm['linkage_acc_test']:.1%}</b>
(gap {rlm['linkage_acc_train'] - rlm['linkage_acc_test']:+.1%}).</div>
<h2>Attack performance by number of shadow models (K)</h2>
<p class="note">AUC 0.5 = random guessing, 1.0 = perfect. A random guesser gets 1% TPR at 1% FPR and 0.1% at 0.1% FPR.
"Shadow AUC" is measured on shadow models the attack never saw; "Target AUC" is the real test.</p>
<table><tr><th>K</th><th>Shadow AUC</th><th>Target AUC</th><th>95% CI</th><th>TPR @ 1% FPR</th>
<th>TPR @ 0.1% FPR</th><th>Balanced acc.</th><th>Verdict</th></tr>{rows}</table>
<h2>Feature-level view (K={k})</h2>
<table><tr><th>Feature</th><th>Single-feature AUC on target</th><th>Gap on shadow (d)</th><th>Gap on target (d)</th></tr>{feat_rows}</table>
<h2>Figures</h2>{figs}
<h2>Limitations</h2><ul>{limits}</ul></body></html>"""


def main():
    if not os.path.exists(RESULTS_PATH):
        raise SystemExit(f"{RESULTS_PATH} not found - run attack_model.py first.")
    with open(RESULTS_PATH) as fh:
        res = json.load(fh)
    print_console(res)
    with open(OUT_HTML, "w", encoding="utf-8") as fh:
        fh.write(build_html(res))
    print(f"\nHTML report written to {OUT_HTML} (open it in your browser).")


if __name__ == "__main__":
    main()
