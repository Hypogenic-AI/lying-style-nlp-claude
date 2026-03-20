# Research Plan: Lying Style in LLM-Generated Text

## Motivation & Novelty Assessment

### Why This Research Matters
Understanding whether LLMs produce detectably different text when lying has direct implications for AI safety, misinformation detection, and trust in AI systems. If lying produces distributional signatures in the output text alone (without access to model internals), it enables black-box detection of deceptive AI outputs — critical for deployed systems where internal states are inaccessible.

### Gap in Existing Work
Most prior work (Azaria & Mitchell 2023, Long et al. 2025, Huan et al. 2025) focuses on **internal representations** (hidden states, activations) to detect lying. Very few studies systematically analyze whether the **output text distribution** itself differs between truthful and deceptive conditions. The literature review identifies "output-level distributional analysis" as explicitly underexplored.

### Our Novel Contribution
We conduct a systematic, black-box analysis of **output text features** when LLMs are instructed to lie vs. respond truthfully. We test multiple deception elicitation strategies (direct instruction, roleplay, jailbreak-style) across multiple models, comparing surface-level linguistic features, vocabulary usage, and text statistics. This fills the gap between internal-representation approaches and practical black-box detection.

### Experiment Justification
- **Experiment 1 (Paired Corpus Generation)**: Generate truthful vs. deceptive responses to identical factual questions using real LLM APIs. Needed to create controlled paired data.
- **Experiment 2 (Distributional Feature Analysis)**: Compare linguistic features (sentence length, vocabulary diversity, hedging, sentiment, readability) between conditions. Tests whether surface-level differences exist.
- **Experiment 3 (Statistical Divergence)**: Compute KL/JS divergence on vocabulary distributions and n-gram statistics. Quantifies distributional distance.
- **Experiment 4 (Classifier-Based Detection)**: Train classifiers on text features to detect lying. Tests practical detectability from output alone.

## Research Question
Is the distribution of text produced by an LLM when instructed to lie noticeably different from its truthful output, and can this difference be detected from the text alone (black-box)?

## Hypothesis Decomposition
1. **H1**: LLMs instructed to lie produce text with different surface-level statistics (sentence length, word count, vocabulary diversity) compared to truthful responses.
2. **H2**: The vocabulary distribution (unigram/bigram frequencies) differs measurably between truthful and deceptive outputs (JS divergence > baseline).
3. **H3**: Deceptive text exhibits different hedging/uncertainty markers (more or fewer hedge words, qualifiers).
4. **H4**: A text-only classifier can distinguish truthful from deceptive LLM output above chance.
5. **H5**: These effects generalize across different deception elicitation methods (direct, roleplay, jailbreak).

## Proposed Methodology

### Approach
1. Select 100 factual questions from TruthfulQA and Azaria datasets
2. Prompt GPT-4.1 under 4 conditions: truthful, direct-lie, roleplay-lie, jailbreak-lie
3. Collect responses (3 runs per condition for variance)
4. Extract linguistic features from all responses
5. Statistical comparison + classifier training

### Baselines
- Random classifier (50%)
- Majority class classifier
- Simple perplexity comparison

### Evaluation Metrics
- Cohen's d effect size for feature differences
- Jensen-Shannon divergence for vocabulary distributions
- Classification accuracy, F1, AUC-ROC for classifier
- Bootstrap confidence intervals

### Statistical Analysis Plan
- Paired t-tests / Wilcoxon signed-rank tests for feature comparisons
- Bonferroni correction for multiple comparisons
- Bootstrap resampling for confidence intervals
- Effect size reporting (Cohen's d) for all comparisons

## Expected Outcomes
- Truthful and deceptive text will show statistically significant differences in some features (especially hedging, sentence length, vocabulary diversity)
- JS divergence will be small but significant (lying doesn't change style dramatically but introduces subtle shifts)
- Classifier accuracy will be above chance (55-70%) but not near-perfect, consistent with prior work showing surface features have limited but real discriminative power

## Timeline
- Phase 1 (Planning): 15 min ✓
- Phase 2 (Setup + Data Generation): 30 min
- Phase 3 (Feature Extraction + Analysis): 30 min
- Phase 4 (Classification + Statistical Testing): 20 min
- Phase 5 (Documentation): 20 min

## Potential Challenges
- API rate limits → use batched requests with retry logic
- LLM may refuse to lie → use multiple elicitation strategies
- Small effect sizes → use adequate sample size and proper statistical tests
- Confounding instruction style with lying style → use multiple prompt templates

## Success Criteria
- Generate paired corpus of ≥100 questions × 4 conditions
- Identify at least 2 features with statistically significant differences (p < 0.05 after correction)
- Classifier achieves >55% accuracy (above chance with p < 0.05)
- Clear visualization of distributional differences
