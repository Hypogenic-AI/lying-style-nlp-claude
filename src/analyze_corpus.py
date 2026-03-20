"""
Analyze distributional differences between truthful and deceptive LLM outputs.
Computes linguistic features, statistical tests, and generates visualizations.
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
from sklearn.metrics import (accuracy_score, classification_report, roc_auc_score,
                              confusion_matrix, f1_score)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Reproducibility
np.random.seed(42)

# Hedging / uncertainty words
HEDGE_WORDS = {
    "maybe", "perhaps", "possibly", "might", "could", "would", "seem",
    "appears", "likely", "unlikely", "suggest", "believe", "think",
    "probably", "approximately", "roughly", "about", "around", "generally",
    "typically", "usually", "often", "sometimes", "arguably", "supposedly",
    "allegedly", "apparently", "presumably", "conceivably", "potentially",
    "uncertain", "unclear", "debatable"
}

# Certainty / assertive words
CERTAINTY_WORDS = {
    "definitely", "certainly", "absolutely", "clearly", "obviously",
    "undoubtedly", "indeed", "surely", "always", "never", "every",
    "must", "proven", "established", "confirmed", "known", "fact",
    "true", "false", "correct", "incorrect", "wrong", "right",
    "no doubt", "without question", "unquestionably"
}

# Negation words
NEGATION_WORDS = {
    "not", "no", "never", "neither", "nor", "none", "nothing",
    "nowhere", "nobody", "isn't", "wasn't", "aren't", "weren't",
    "don't", "doesn't", "didn't", "won't", "wouldn't", "shouldn't",
    "couldn't", "can't", "cannot"
}


def extract_features(text):
    """Extract linguistic features from a text response."""
    if not text or not isinstance(text, str):
        return None

    words = text.lower().split()
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    tokens = re.findall(r'\b\w+\b', text.lower())

    if len(tokens) == 0:
        return None

    # Basic statistics
    word_count = len(tokens)
    sentence_count = max(len(sentences), 1)
    avg_word_length = np.mean([len(w) for w in tokens])
    avg_sentence_length = word_count / sentence_count

    # Vocabulary diversity
    unique_tokens = set(tokens)
    type_token_ratio = len(unique_tokens) / word_count
    hapax_legomena = sum(1 for w, c in Counter(tokens).items() if c == 1)
    hapax_ratio = hapax_legomena / word_count

    # Hedging & certainty
    hedge_count = sum(1 for w in tokens if w in HEDGE_WORDS)
    certainty_count = sum(1 for w in tokens if w in CERTAINTY_WORDS)
    hedge_ratio = hedge_count / word_count
    certainty_ratio = certainty_count / word_count

    # Negation
    negation_count = sum(1 for w in tokens if w in NEGATION_WORDS)
    negation_ratio = negation_count / word_count

    # Punctuation
    exclamation_count = text.count("!")
    question_count = text.count("?")
    comma_count = text.count(",")
    punctuation_ratio = sum(1 for c in text if c in string.punctuation) / max(len(text), 1)

    # Capitalization
    upper_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)

    # Response starts with "True" or "False" assessment
    starts_true = 1 if re.match(r'\s*(true|yes|correct)', text.lower()) else 0
    starts_false = 1 if re.match(r'\s*(false|no|incorrect)', text.lower()) else 0

    # N-gram features (bigram diversity)
    bigrams = list(zip(tokens[:-1], tokens[1:]))
    bigram_diversity = len(set(bigrams)) / max(len(bigrams), 1)

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_word_length": avg_word_length,
        "avg_sentence_length": avg_sentence_length,
        "type_token_ratio": type_token_ratio,
        "hapax_ratio": hapax_ratio,
        "hedge_ratio": hedge_ratio,
        "certainty_ratio": certainty_ratio,
        "negation_ratio": negation_ratio,
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "comma_count": comma_count,
        "punctuation_ratio": punctuation_ratio,
        "upper_ratio": upper_ratio,
        "starts_true": starts_true,
        "starts_false": starts_false,
        "bigram_diversity": bigram_diversity,
    }


def compute_js_divergence(texts_a, texts_b, ngram=1):
    """Compute Jensen-Shannon divergence between two sets of texts."""
    def get_ngram_dist(texts, n):
        counter = Counter()
        for text in texts:
            tokens = re.findall(r'\b\w+\b', text.lower())
            if n == 1:
                counter.update(tokens)
            else:
                counter.update(zip(*[tokens[i:] for i in range(n)]))
        total = sum(counter.values())
        return {k: v/total for k, v in counter.items()}

    dist_a = get_ngram_dist(texts_a, ngram)
    dist_b = get_ngram_dist(texts_b, ngram)

    # Get common vocabulary
    vocab = sorted(set(list(dist_a.keys()) + list(dist_b.keys())))
    p = np.array([dist_a.get(w, 0) for w in vocab])
    q = np.array([dist_b.get(w, 0) for w in vocab])

    # Add small epsilon to avoid log(0)
    eps = 1e-10
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()

    return jensenshannon(p, q) ** 2  # squared JS divergence


def run_analysis(corpus_path="results/paired_corpus.json"):
    """Run full distributional analysis."""
    print("=" * 60)
    print("Analyzing Paired Truthful/Deceptive Corpus")
    print("=" * 60)

    # Load corpus
    with open(corpus_path) as f:
        corpus = json.load(f)

    print(f"Loaded {len(corpus)} responses")

    # Extract features
    records = []
    for item in corpus:
        feats = extract_features(item["response"])
        if feats is None:
            continue
        feats["condition"] = item["condition"]
        feats["question_id"] = item["question_id"]
        feats["run"] = item["run"]
        feats["source"] = item["source"]
        feats["is_lying"] = 0 if item["condition"] == "truthful" else 1
        records.append(feats)

    df = pd.DataFrame(records)
    print(f"Extracted features for {len(df)} responses")
    print(f"Conditions: {df['condition'].value_counts().to_dict()}")

    # Save features
    df.to_csv("results/features.csv", index=False)

    # =========================================================================
    # 1. Descriptive Statistics by Condition
    # =========================================================================
    print("\n" + "=" * 60)
    print("1. DESCRIPTIVE STATISTICS")
    print("=" * 60)

    feature_cols = [c for c in df.columns if c not in
                    ["condition", "question_id", "run", "source", "is_lying"]]

    desc_stats = df.groupby("condition")[feature_cols].agg(["mean", "std"])
    print(desc_stats.round(4).to_string())
    desc_stats.to_csv("results/descriptive_stats.csv")

    # =========================================================================
    # 2. Statistical Tests: Truthful vs Each Lying Condition
    # =========================================================================
    print("\n" + "=" * 60)
    print("2. STATISTICAL TESTS (Truthful vs Lying)")
    print("=" * 60)

    truthful = df[df["condition"] == "truthful"]
    lying_conditions = ["direct_lie", "roleplay_lie", "sycophantic_lie"]

    test_results = []
    for cond in lying_conditions:
        lying = df[df["condition"] == cond]
        print(f"\n--- Truthful vs {cond} ---")
        for feat in feature_cols:
            t_vals = truthful[feat].dropna()
            l_vals = lying[feat].dropna()

            # Mann-Whitney U test (non-parametric, no normality assumption)
            u_stat, p_val = stats.mannwhitneyu(t_vals, l_vals, alternative="two-sided")

            # Cohen's d effect size
            pooled_std = np.sqrt((t_vals.std()**2 + l_vals.std()**2) / 2)
            cohens_d = (t_vals.mean() - l_vals.mean()) / pooled_std if pooled_std > 0 else 0

            test_results.append({
                "comparison": f"truthful_vs_{cond}",
                "feature": feat,
                "truthful_mean": t_vals.mean(),
                "lying_mean": l_vals.mean(),
                "u_statistic": u_stat,
                "p_value": p_val,
                "cohens_d": cohens_d,
                "significant_005": p_val < 0.05,
                "significant_bonferroni": p_val < 0.05 / (len(feature_cols) * len(lying_conditions))
            })

    test_df = pd.DataFrame(test_results)
    test_df.to_csv("results/statistical_tests.csv", index=False)

    # Print significant results
    sig = test_df[test_df["significant_bonferroni"]]
    print(f"\nSignificant results (Bonferroni-corrected, α = {0.05 / (len(feature_cols) * len(lying_conditions)):.6f}):")
    if len(sig) > 0:
        for _, row in sig.iterrows():
            print(f"  {row['comparison']}: {row['feature']} "
                  f"(d={row['cohens_d']:.3f}, p={row['p_value']:.2e})")
    else:
        print("  No results survived Bonferroni correction.")

    # Also show uncorrected significant results
    sig_uncorrected = test_df[test_df["significant_005"]]
    print(f"\nSignificant at p < 0.05 (uncorrected): {len(sig_uncorrected)} / {len(test_df)}")

    # =========================================================================
    # 3. Overall Truthful vs All Lying
    # =========================================================================
    print("\n" + "=" * 60)
    print("3. OVERALL: TRUTHFUL vs ALL LYING CONDITIONS")
    print("=" * 60)

    all_lying = df[df["is_lying"] == 1]
    overall_results = []
    for feat in feature_cols:
        t_vals = truthful[feat].dropna()
        l_vals = all_lying[feat].dropna()
        u_stat, p_val = stats.mannwhitneyu(t_vals, l_vals, alternative="two-sided")
        pooled_std = np.sqrt((t_vals.std()**2 + l_vals.std()**2) / 2)
        cohens_d = (t_vals.mean() - l_vals.mean()) / pooled_std if pooled_std > 0 else 0
        overall_results.append({
            "feature": feat,
            "truthful_mean": t_vals.mean(),
            "lying_mean": l_vals.mean(),
            "diff": t_vals.mean() - l_vals.mean(),
            "cohens_d": cohens_d,
            "p_value": p_val,
            "significant": p_val < 0.05 / len(feature_cols)
        })
        if p_val < 0.05:
            print(f"  {feat}: truth={t_vals.mean():.4f}, lie={l_vals.mean():.4f}, "
                  f"d={cohens_d:.3f}, p={p_val:.2e} {'***' if p_val < 0.05/len(feature_cols) else '*'}")

    overall_df = pd.DataFrame(overall_results)
    overall_df.to_csv("results/overall_tests.csv", index=False)

    # =========================================================================
    # 4. Jensen-Shannon Divergence
    # =========================================================================
    print("\n" + "=" * 60)
    print("4. JENSEN-SHANNON DIVERGENCE")
    print("=" * 60)

    # Get texts directly from the original corpus data
    truthful_texts = [item["response"] for item in corpus if item["condition"] == "truthful"]

    js_results = []
    for cond in lying_conditions:
        lying_texts = [item["response"] for item in corpus if item["condition"] == cond]
        for ngram in [1, 2]:
            js_div = compute_js_divergence(truthful_texts, lying_texts, ngram=ngram)
            js_results.append({
                "comparison": f"truthful_vs_{cond}",
                "ngram": ngram,
                "js_divergence": js_div
            })
            print(f"  Truthful vs {cond} ({ngram}-gram): JSD = {js_div:.6f}")

    # Baseline: compare truthful run 0 vs truthful runs 1,2 (expected small divergence)
    t_run0 = [item["response"] for item in corpus if item["condition"] == "truthful" and item["run"] == 0]
    t_run12 = [item["response"] for item in corpus if item["condition"] == "truthful" and item["run"] in (1, 2)]
    for ngram in [1, 2]:
        baseline_js = compute_js_divergence(t_run0, t_run12, ngram=ngram)
        js_results.append({
            "comparison": "truthful_run0_vs_runs12 (baseline)",
            "ngram": ngram,
            "js_divergence": baseline_js
        })
        print(f"  Truthful self-divergence baseline ({ngram}-gram): JSD = {baseline_js:.6f}")

    js_df = pd.DataFrame(js_results)
    js_df.to_csv("results/js_divergence.csv", index=False)

    # =========================================================================
    # 5. Classification: Can we detect lying from text features alone?
    # =========================================================================
    print("\n" + "=" * 60)
    print("5. CLASSIFICATION")
    print("=" * 60)

    X = df[feature_cols].values
    y = df["is_lying"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    classifiers = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf_results = []

    for name, clf in classifiers.items():
        scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="accuracy")
        f1_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="f1")
        auc_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="roc_auc")

        clf_results.append({
            "classifier": name,
            "accuracy_mean": scores.mean(),
            "accuracy_std": scores.std(),
            "f1_mean": f1_scores.mean(),
            "f1_std": f1_scores.std(),
            "auc_mean": auc_scores.mean(),
            "auc_std": auc_scores.std()
        })
        print(f"  {name}: Acc={scores.mean():.3f}±{scores.std():.3f}, "
              f"F1={f1_scores.mean():.3f}±{f1_scores.std():.3f}, "
              f"AUC={auc_scores.mean():.3f}±{auc_scores.std():.3f}")

    clf_df = pd.DataFrame(clf_results)
    clf_df.to_csv("results/classification_results.csv", index=False)

    # Feature importance from Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)
    importances = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)
    importances.to_csv("results/feature_importances.csv", index=False)
    print(f"\nTop 5 features for classification:")
    for _, row in importances.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    # =========================================================================
    # 6. Per-condition classification (which lying strategy is most detectable?)
    # =========================================================================
    print("\n" + "=" * 60)
    print("6. PER-CONDITION DETECTION")
    print("=" * 60)

    per_cond_results = []
    for cond in lying_conditions:
        subset = df[(df["condition"] == "truthful") | (df["condition"] == cond)]
        X_sub = subset[feature_cols].values
        y_sub = (subset["condition"] != "truthful").astype(int).values

        X_sub_scaled = scaler.fit_transform(X_sub)
        clf = LogisticRegression(random_state=42, max_iter=1000)
        scores = cross_val_score(clf, X_sub_scaled, y_sub, cv=cv, scoring="accuracy")
        auc_scores = cross_val_score(clf, X_sub_scaled, y_sub, cv=cv, scoring="roc_auc")

        per_cond_results.append({
            "condition": cond,
            "accuracy_mean": scores.mean(),
            "accuracy_std": scores.std(),
            "auc_mean": auc_scores.mean(),
            "auc_std": auc_scores.std()
        })
        print(f"  Truthful vs {cond}: Acc={scores.mean():.3f}±{scores.std():.3f}, "
              f"AUC={auc_scores.mean():.3f}±{auc_scores.std():.3f}")

    per_cond_df = pd.DataFrame(per_cond_results)
    per_cond_df.to_csv("results/per_condition_classification.csv", index=False)

    # =========================================================================
    # 7. Visualizations
    # =========================================================================
    print("\n" + "=" * 60)
    print("7. GENERATING VISUALIZATIONS")
    print("=" * 60)

    plot_dir = Path("results/plots")
    plot_dir.mkdir(exist_ok=True)

    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)

    # 7a. Feature distributions by condition
    key_features = ["word_count", "avg_sentence_length", "type_token_ratio",
                     "hedge_ratio", "certainty_ratio", "negation_ratio"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for idx, feat in enumerate(key_features):
        ax = axes[idx // 3][idx % 3]
        for cond in ["truthful", "direct_lie", "roleplay_lie", "sycophantic_lie"]:
            vals = df[df["condition"] == cond][feat]
            ax.hist(vals, bins=20, alpha=0.5, label=cond, density=True)
        ax.set_title(feat)
        ax.legend(fontsize=8)
    plt.suptitle("Feature Distributions by Condition", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(plot_dir / "feature_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 7b. Effect sizes heatmap
    effect_sizes = test_df.pivot(index="feature", columns="comparison", values="cohens_d")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(effect_sizes, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Cohen's d Effect Sizes: Truthful vs Lying Conditions")
    plt.tight_layout()
    plt.savefig(plot_dir / "effect_sizes_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 7c. JS Divergence comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    js_plot = js_df[~js_df["comparison"].str.contains("baseline")]
    for ngram in [1, 2]:
        subset = js_plot[js_plot["ngram"] == ngram]
        labels = [c.replace("truthful_vs_", "") for c in subset["comparison"]]
        ax.bar([f"{l}\n({ngram}-gram)" for l in labels],
               subset["js_divergence"].values,
               alpha=0.7, label=f"{ngram}-gram")

    # Add baseline line
    baseline_vals = js_df[js_df["comparison"].str.contains("baseline")]
    for _, row in baseline_vals.iterrows():
        ax.axhline(y=row["js_divergence"], color="red", linestyle="--",
                   alpha=0.5, label=f"Baseline ({row['ngram']}-gram)" if row['ngram'] == 1 else "")
    ax.set_ylabel("Jensen-Shannon Divergence")
    ax.set_title("Vocabulary Distribution Divergence: Truthful vs Lying")
    ax.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "js_divergence.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 7d. Classification performance comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(clf_df))
    ax.bar(x, clf_df["accuracy_mean"], yerr=clf_df["accuracy_std"],
           capsize=5, alpha=0.7, color="steelblue")
    ax.axhline(y=0.5, color="red", linestyle="--", label="Chance (50%)")
    ax.axhline(y=0.75, color="green", linestyle="--", alpha=0.5, label="75% baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(clf_df["classifier"], rotation=15)
    ax.set_ylabel("Accuracy")
    ax.set_title("Classification: Truthful vs Lying (5-fold CV)")
    ax.legend()
    ax.set_ylim(0.4, 1.0)
    plt.tight_layout()
    plt.savefig(plot_dir / "classification_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 7e. Feature importance
    fig, ax = plt.subplots(figsize=(8, 6))
    top_feats = importances.head(10)
    ax.barh(range(len(top_feats)), top_feats["importance"].values, color="steelblue")
    ax.set_yticks(range(len(top_feats)))
    ax.set_yticklabels(top_feats["feature"].values)
    ax.set_xlabel("Feature Importance (Random Forest)")
    ax.set_title("Top 10 Features for Detecting Lying")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(plot_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 7f. Box plots of key features
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for idx, feat in enumerate(key_features):
        ax = axes[idx // 3][idx % 3]
        data = [df[df["condition"] == c][feat].values for c in
                ["truthful", "direct_lie", "roleplay_lie", "sycophantic_lie"]]
        bp = ax.boxplot(data, labels=["truthful", "direct", "roleplay", "sycophantic"],
                        patch_artist=True)
        colors = ["#4CAF50", "#F44336", "#FF9800", "#9C27B0"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        ax.set_title(feat)
        ax.tick_params(axis='x', rotation=30)
    plt.suptitle("Feature Distributions: Truthful vs Lying Conditions", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(plot_dir / "feature_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Visualizations saved to results/plots/")

    # =========================================================================
    # 8. Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    n_sig_bonferroni = len(test_df[test_df["significant_bonferroni"]])
    n_sig_uncorrected = len(test_df[test_df["significant_005"]])
    best_clf = clf_df.loc[clf_df["accuracy_mean"].idxmax()]

    summary = {
        "total_responses": len(df),
        "conditions": df["condition"].value_counts().to_dict(),
        "significant_features_bonferroni": n_sig_bonferroni,
        "significant_features_uncorrected": n_sig_uncorrected,
        "total_tests": len(test_df),
        "best_classifier": best_clf["classifier"],
        "best_accuracy": f"{best_clf['accuracy_mean']:.3f}±{best_clf['accuracy_std']:.3f}",
        "best_auc": f"{best_clf['auc_mean']:.3f}±{best_clf['auc_std']:.3f}",
        "js_divergences": {row["comparison"]: row["js_divergence"]
                          for _, row in js_df.iterrows()},
        "top_discriminative_features": importances.head(5)["feature"].tolist(),
        "overall_significant_features": overall_df[overall_df["significant"]]["feature"].tolist()
    }

    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))

    return summary


if __name__ == "__main__":
    run_analysis()
