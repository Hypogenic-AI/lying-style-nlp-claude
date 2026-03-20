# Lying Style: Distributional Differences in LLM Deceptive Text

## Overview
This project investigates whether LLMs produce detectably different text when instructed to lie compared to truthful responses. We generated 1,344 paired responses from GPT-4.1 and GPT-4o-mini under truthful and deceptive conditions, then analyzed distributional differences using linguistic features, statistical tests, and machine learning classifiers.

## Key Findings
- **Yes, lying text is noticeably different.** 9 of 15 linguistic features show statistically significant differences (Bonferroni-corrected, p < 0.003).
- **Negation is the strongest signal**: Lying responses use 7x more negation words (Cohen's d = -1.23).
- **Lying responses are longer**: +10 words on average (d = -1.08) with longer sentences (d = -1.04).
- **Hedging increases 2.8x** when lying (d = -0.50).
- **92.8% classification accuracy** using only surface-level text features (Random Forest, 5-fold CV).
- **Results generalize** across 3 deception strategies and 2 models.

## How to Reproduce

```bash
# Setup
uv venv && source .venv/bin/activate
uv add openai numpy pandas matplotlib scikit-learn scipy seaborn tqdm nltk textstat datasets pyarrow

# Generate corpus (requires OPENAI_API_KEY)
python src/generate_corpus.py          # GPT-4.1, ~15 min
python src/generate_corpus_model2.py   # GPT-4o-mini, ~5 min

# Run analysis
python src/analyze_corpus.py           # Main analysis
python src/analyze_robust.py           # Robust analysis with bootstrap CIs
```

## File Structure
```
├── REPORT.md                    # Full research report with results
├── planning.md                  # Research plan
├── src/
│   ├── generate_corpus.py       # GPT-4.1 corpus generation
│   ├── generate_corpus_model2.py # GPT-4o-mini corpus generation
│   ├── analyze_corpus.py        # Main analysis pipeline
│   └── analyze_robust.py        # Robust analysis with bootstrap CIs
├── results/
│   ├── paired_corpus.json       # GPT-4.1 responses (1,152)
│   ├── paired_corpus_gpt4omini.json  # GPT-4o-mini responses (192)
│   ├── features.csv             # Extracted features
│   ├── statistical_tests.csv    # All statistical test results
│   ├── classification_results.csv
│   ├── summary.json
│   └── plots/                   # Visualizations
├── datasets/                    # Source datasets
├── papers/                      # Downloaded research papers
└── literature_review.md         # Literature review
```

See [REPORT.md](REPORT.md) for full details.
