"""
Robust analysis: exclude trivial features (starts_true/starts_false) and
analyze whether deeper linguistic features still distinguish truthful from deceptive text.
Also: cross-model analysis with GPT-4o-mini data.
"""

import json
import re
import string
import warnings
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
np.random.seed(42)

# Same feature extraction as main analysis
HEDGE_WORDS = {
    "maybe", "perhaps", "possibly", "might", "could", "would", "seem",
    "appears", "likely", "unlikely", "suggest", "believe", "think",
    "probably", "approximately", "roughly", "about", "around", "generally",
    "typically", "usually", "often", "sometimes", "arguably", "supposedly",
    "allegedly", "apparently", "presumably", "conceivably", "potentially",
    "uncertain", "unclear", "debatable"
}
CERTAINTY_WORDS = {
    "definitely", "certainly", "absolutely", "clearly", "obviously",
    "undoubtedly", "indeed", "surely", "always", "never", "every",
    "must", "proven", "established", "confirmed", "known", "fact",
    "true", "false", "correct", "incorrect", "wrong", "right",
    "no doubt", "without question", "unquestionably"
}
NEGATION_WORDS = {
    "not", "no", "never", "neither", "nor", "none", "nothing",
    "nowhere", "nobody", "isn't", "wasn't", "aren't", "weren't",
    "don't", "doesn't", "didn't", "won't", "wouldn't", "shouldn't",
    "couldn't", "can't", "cannot"
}


def extract_features(text):
    if not text or not isinstance(text, str):
        return None
    tokens = re.findall(r'\b\w+\b', text.lower())
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if len(tokens) == 0:
        return None

    word_count = len(tokens)
    sentence_count = max(len(sentences), 1)

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_word_length": np.mean([len(w) for w in tokens]),
        "avg_sentence_length": word_count / sentence_count,
        "type_token_ratio": len(set(tokens)) / word_count,
        "hapax_ratio": sum(1 for w, c in Counter(tokens).items() if c == 1) / word_count,
        "hedge_ratio": sum(1 for w in tokens if w in HEDGE_WORDS) / word_count,
        "certainty_ratio": sum(1 for w in tokens if w in CERTAINTY_WORDS) / word_count,
        "negation_ratio": sum(1 for w in tokens if w in NEGATION_WORDS) / word_count,
        "exclamation_count": text.count("!"),
        "question_count": text.count("?"),
        "comma_count": text.count(","),
        "punctuation_ratio": sum(1 for c in text if c in string.punctuation) / max(len(text), 1),
        "upper_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1),
        "bigram_diversity": len(set(zip(tokens[:-1], tokens[1:]))) / max(len(tokens)-1, 1),
    }


def analyze_model(corpus_path, model_name, output_prefix):
    """Run analysis for one model's corpus."""
    print(f"\n{'='*60}")
    print(f"ANALYSIS: {model_name}")
    print(f"{'='*60}")

    with open(corpus_path) as f:
        corpus = json.load(f)

    # Extract features
    records = []
    for item in corpus:
        feats = extract_features(item["response"])
        if feats is None:
            continue
        feats["condition"] = item["condition"]
        feats["question_id"] = item["question_id"]
        feats["is_lying"] = 0 if item["condition"] == "truthful" else 1
        records.append(feats)

    df = pd.DataFrame(records)
    print(f"Responses: {len(df)}, Conditions: {df['condition'].value_counts().to_dict()}")

    # Exclude trivial features for robust analysis
    feature_cols = [c for c in df.columns if c not in
                    ["condition", "question_id", "is_lying", "run", "source"]]

    # =====================
    # Statistical tests (overall: truthful vs all lying)
    # =====================
    truthful = df[df["is_lying"] == 0]
    lying = df[df["is_lying"] == 1]

    print(f"\n--- Overall: Truthful vs All Lying ---")
    results = []
    for feat in feature_cols:
        t_vals = truthful[feat].dropna()
        l_vals = lying[feat].dropna()
        u_stat, p_val = stats.mannwhitneyu(t_vals, l_vals, alternative="two-sided")
        pooled_std = np.sqrt((t_vals.std()**2 + l_vals.std()**2) / 2)
        d = (t_vals.mean() - l_vals.mean()) / pooled_std if pooled_std > 0 else 0
        sig = p_val < 0.05 / len(feature_cols)
        results.append({
            "feature": feat, "truth_mean": t_vals.mean(), "lie_mean": l_vals.mean(),
            "cohens_d": d, "p_value": p_val, "bonferroni_sig": sig
        })
        if sig:
            print(f"  {feat}: truth={t_vals.mean():.4f} lie={l_vals.mean():.4f} d={d:.3f} p={p_val:.2e} ***")

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"results/{output_prefix}_stats.csv", index=False)

    # =====================
    # Classification (without trivial starts_true/starts_false)
    # =====================
    X = df[feature_cols].values
    y = df["is_lying"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"\n--- Classification (all features) ---")
    clf_results = {}
    for name, clf in [
        ("LogReg", LogisticRegression(random_state=42, max_iter=1000)),
        ("RF", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("GB", GradientBoostingClassifier(n_estimators=100, random_state=42))
    ]:
        acc = cross_val_score(clf, X_scaled, y, cv=cv, scoring="accuracy")
        auc = cross_val_score(clf, X_scaled, y, cv=cv, scoring="roc_auc")
        clf_results[name] = {"acc": acc.mean(), "acc_std": acc.std(),
                              "auc": auc.mean(), "auc_std": auc.std()}
        print(f"  {name}: Acc={acc.mean():.3f}±{acc.std():.3f} AUC={auc.mean():.3f}±{auc.std():.3f}")

    # Feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)
    importances = sorted(zip(feature_cols, rf.feature_importances_),
                         key=lambda x: x[1], reverse=True)
    print(f"\n--- Top features ---")
    for feat, imp in importances[:7]:
        print(f"  {feat}: {imp:.4f}")

    return df, results_df, clf_results, importances


def compute_js_divergence(texts_a, texts_b, ngram=1):
    def get_dist(texts, n):
        counter = Counter()
        for text in texts:
            tokens = re.findall(r'\b\w+\b', text.lower())
            if n == 1:
                counter.update(tokens)
            else:
                counter.update(zip(*[tokens[i:] for i in range(n)]))
        total = sum(counter.values())
        return {k: v/total for k, v in counter.items()}

    dist_a = get_dist(texts_a, ngram)
    dist_b = get_dist(texts_b, ngram)
    vocab = sorted(set(list(dist_a.keys()) + list(dist_b.keys())))
    p = np.array([dist_a.get(w, 0) for w in vocab]) + 1e-10
    q = np.array([dist_b.get(w, 0) for w in vocab]) + 1e-10
    p /= p.sum()
    q /= q.sum()
    return jensenshannon(p, q) ** 2


def bootstrap_test(values_a, values_b, n_bootstrap=10000):
    """Bootstrap confidence interval for difference in means."""
    observed_diff = np.mean(values_a) - np.mean(values_b)
    combined = np.concatenate([values_a, values_b])
    n_a = len(values_a)
    diffs = []
    for _ in range(n_bootstrap):
        perm = np.random.permutation(combined)
        diffs.append(np.mean(perm[:n_a]) - np.mean(perm[n_a:]))
    diffs = np.array(diffs)
    p_value = np.mean(np.abs(diffs) >= np.abs(observed_diff))
    ci_low = np.percentile(diffs, 2.5)
    ci_high = np.percentile(diffs, 97.5)
    return observed_diff, p_value, ci_low, ci_high


def main():
    plot_dir = Path("results/plots")
    plot_dir.mkdir(exist_ok=True)
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)

    # Analyze GPT-4.1 corpus
    df_gpt41, stats_gpt41, clf_gpt41, imp_gpt41 = analyze_model(
        "results/paired_corpus.json", "GPT-4.1", "gpt41")

    # Analyze GPT-4o-mini corpus (if exists)
    df_mini = None
    mini_path = Path("results/paired_corpus_gpt4omini.json")
    if mini_path.exists():
        df_mini, stats_mini, clf_mini, imp_mini = analyze_model(
            str(mini_path), "GPT-4o-mini", "gpt4omini")

    # =====================
    # Cross-model comparison
    # =====================
    if df_mini is not None:
        print(f"\n{'='*60}")
        print("CROSS-MODEL COMPARISON")
        print(f"{'='*60}")

        # Compare effect sizes across models
        merge_cols = ["feature"]
        comparison = stats_gpt41[["feature", "cohens_d", "bonferroni_sig"]].rename(
            columns={"cohens_d": "d_gpt41", "bonferroni_sig": "sig_gpt41"})
        comparison = comparison.merge(
            stats_mini[["feature", "cohens_d", "bonferroni_sig"]].rename(
                columns={"cohens_d": "d_gpt4omini", "bonferroni_sig": "sig_gpt4omini"}),
            on="feature")

        print("\nEffect size comparison:")
        for _, row in comparison.iterrows():
            if row["sig_gpt41"] or row["sig_gpt4omini"]:
                print(f"  {row['feature']}: GPT-4.1 d={row['d_gpt41']:.3f} "
                      f"GPT-4o-mini d={row['d_gpt4omini']:.3f}")

        comparison.to_csv("results/cross_model_comparison.csv", index=False)

        # Plot cross-model effect sizes
        fig, ax = plt.subplots(figsize=(10, 8))
        sig_feats = comparison[comparison["sig_gpt41"] | comparison["sig_gpt4omini"]]
        x = range(len(sig_feats))
        width = 0.35
        ax.barh([i - width/2 for i in x], sig_feats["d_gpt41"].abs(),
                height=width, label="GPT-4.1", alpha=0.8, color="steelblue")
        ax.barh([i + width/2 for i in x], sig_feats["d_gpt4omini"].abs(),
                height=width, label="GPT-4o-mini", alpha=0.8, color="coral")
        ax.set_yticks(list(x))
        ax.set_yticklabels(sig_feats["feature"].values)
        ax.set_xlabel("|Cohen's d|")
        ax.set_title("Effect Sizes: Truthful vs Lying (Cross-Model)")
        ax.legend()
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(plot_dir / "cross_model_effect_sizes.png", dpi=150, bbox_inches="tight")
        plt.close()

    # =====================
    # Bootstrap analysis for key features (GPT-4.1)
    # =====================
    print(f"\n{'='*60}")
    print("BOOTSTRAP CONFIDENCE INTERVALS (GPT-4.1)")
    print(f"{'='*60}")

    truthful = df_gpt41[df_gpt41["is_lying"] == 0]
    lying = df_gpt41[df_gpt41["is_lying"] == 1]

    key_feats = ["word_count", "avg_sentence_length", "hedge_ratio",
                 "negation_ratio", "type_token_ratio", "upper_ratio"]
    bootstrap_results = []
    for feat in key_feats:
        t_vals = truthful[feat].values
        l_vals = lying[feat].values
        diff, p_val, ci_low, ci_high = bootstrap_test(t_vals, l_vals, n_bootstrap=10000)
        bootstrap_results.append({
            "feature": feat, "observed_diff": diff, "p_value": p_val,
            "ci_low": ci_low, "ci_high": ci_high
        })
        print(f"  {feat}: diff={diff:.4f}, p={p_val:.4f}, 95% CI=[{ci_low:.4f}, {ci_high:.4f}]")

    pd.DataFrame(bootstrap_results).to_csv("results/bootstrap_results.csv", index=False)

    # =====================
    # JS Divergence with bootstrap CIs
    # =====================
    print(f"\n{'='*60}")
    print("JS DIVERGENCE WITH BOOTSTRAP CIs")
    print(f"{'='*60}")

    with open("results/paired_corpus.json") as f:
        corpus = json.load(f)

    conditions = ["direct_lie", "roleplay_lie", "sycophantic_lie"]
    truthful_texts = [item["response"] for item in corpus if item["condition"] == "truthful"]

    js_bootstrap = []
    for cond in conditions:
        lying_texts = [item["response"] for item in corpus if item["condition"] == cond]

        # Bootstrap JS divergence
        js_samples = []
        for _ in range(100):
            idx_t = np.random.choice(len(truthful_texts), len(truthful_texts), replace=True)
            idx_l = np.random.choice(len(lying_texts), len(lying_texts), replace=True)
            t_sample = [truthful_texts[i] for i in idx_t]
            l_sample = [lying_texts[i] for i in idx_l]
            js_samples.append(compute_js_divergence(t_sample, l_sample, ngram=1))

        js_mean = np.mean(js_samples)
        js_ci = np.percentile(js_samples, [2.5, 97.5])
        js_bootstrap.append({
            "comparison": f"truthful_vs_{cond}",
            "js_mean": js_mean,
            "js_ci_low": js_ci[0],
            "js_ci_high": js_ci[1]
        })
        print(f"  Truthful vs {cond}: JSD={js_mean:.4f} [{js_ci[0]:.4f}, {js_ci[1]:.4f}]")

    # Baseline self-divergence
    t_run0 = [item["response"] for item in corpus if item["condition"] == "truthful" and item["run"] == 0]
    t_run12 = [item["response"] for item in corpus if item["condition"] == "truthful" and item["run"] in (1, 2)]
    baseline_js = compute_js_divergence(t_run0, t_run12, ngram=1)
    print(f"  Baseline self-divergence: JSD={baseline_js:.4f}")

    pd.DataFrame(js_bootstrap).to_csv("results/js_divergence_bootstrap.csv", index=False)

    # =====================
    # Qualitative examples
    # =====================
    print(f"\n{'='*60}")
    print("QUALITATIVE EXAMPLES")
    print(f"{'='*60}")

    examples = []
    for q_id in [0, 10, 20, 30, 40]:
        q_items = [item for item in corpus if item["question_id"] == q_id and item["run"] == 0]
        if q_items:
            example = {"statement": q_items[0]["statement"]}
            for item in q_items:
                example[item["condition"]] = item["response"]
            examples.append(example)
            print(f"\nStatement: {example['statement']}")
            for cond in ["truthful", "direct_lie", "roleplay_lie", "sycophantic_lie"]:
                if cond in example:
                    print(f"  [{cond}]: {example[cond][:120]}...")

    with open("results/qualitative_examples.json", "w") as f:
        json.dump(examples, f, indent=2)

    # =====================
    # Final summary visualization
    # =====================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Effect sizes for key features
    ax = axes[0][0]
    key_results = stats_gpt41[stats_gpt41["bonferroni_sig"]].sort_values("cohens_d")
    ax.barh(range(len(key_results)), key_results["cohens_d"].values, color="steelblue", alpha=0.8)
    ax.set_yticks(range(len(key_results)))
    ax.set_yticklabels(key_results["feature"].values, fontsize=9)
    ax.set_xlabel("Cohen's d")
    ax.set_title("Effect Sizes: Truthful vs Lying (GPT-4.1)")
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)

    # Plot 2: Mean feature values
    ax = axes[0][1]
    feats_to_plot = ["word_count", "avg_sentence_length", "hedge_ratio", "negation_ratio"]
    truth_means = [stats_gpt41[stats_gpt41["feature"]==f]["truth_mean"].values[0] for f in feats_to_plot]
    lie_means = [stats_gpt41[stats_gpt41["feature"]==f]["lie_mean"].values[0] for f in feats_to_plot]
    x = np.arange(len(feats_to_plot))
    ax.bar(x - 0.2, truth_means, 0.4, label="Truthful", color="#4CAF50", alpha=0.8)
    ax.bar(x + 0.2, lie_means, 0.4, label="Lying", color="#F44336", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(feats_to_plot, rotation=30, fontsize=9)
    ax.set_title("Key Feature Means")
    ax.legend()

    # Plot 3: JS Divergence
    ax = axes[1][0]
    js_data = pd.DataFrame(js_bootstrap)
    ax.bar(range(len(js_data)), js_data["js_mean"],
           yerr=[js_data["js_mean"]-js_data["js_ci_low"], js_data["js_ci_high"]-js_data["js_mean"]],
           capsize=5, alpha=0.8, color="steelblue")
    ax.axhline(y=baseline_js, color="red", linestyle="--", label=f"Baseline ({baseline_js:.3f})")
    ax.set_xticks(range(len(js_data)))
    ax.set_xticklabels([c.replace("truthful_vs_", "") for c in js_data["comparison"]], fontsize=9)
    ax.set_ylabel("JS Divergence")
    ax.set_title("Vocabulary Distribution Divergence")
    ax.legend()

    # Plot 4: Classification accuracy
    ax = axes[1][1]
    models_data = {"GPT-4.1": clf_gpt41}
    if df_mini is not None:
        models_data["GPT-4o-mini"] = clf_mini

    colors = {"GPT-4.1": "steelblue", "GPT-4o-mini": "coral"}
    for i, (model_name, clf_res) in enumerate(models_data.items()):
        clfs = list(clf_res.keys())
        accs = [clf_res[c]["acc"] for c in clfs]
        stds = [clf_res[c]["acc_std"] for c in clfs]
        x_pos = np.arange(len(clfs)) + i * 0.3
        ax.bar(x_pos, accs, 0.25, yerr=stds, capsize=3,
               label=model_name, color=colors.get(model_name, "gray"), alpha=0.8)

    ax.axhline(y=0.5, color="red", linestyle="--", label="Chance")
    ax.set_xticks(np.arange(len(clfs)) + 0.15)
    ax.set_xticklabels(clfs)
    ax.set_ylabel("Accuracy")
    ax.set_title("Classification: Truthful vs Lying")
    ax.set_ylim(0.4, 1.05)
    ax.legend(fontsize=9)

    plt.suptitle("Lying Style Detection: Distributional Analysis", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(plot_dir / "summary_figure.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nAll results saved to results/")
    print("Done!")


if __name__ == "__main__":
    main()
