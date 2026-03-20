# Resources Catalog

## Summary
This document catalogs all resources gathered for the "Lying Style" research project investigating whether LLM text distributions differ under truthful vs deceptive instructions.

---

## Papers
Total papers downloaded: 30

| # | Title | Year | File | Key Info |
|---|-------|------|------|----------|
| 1 | The Internal State of an LLM Knows When It's Lying | 2023 | papers/2304.13734_*.pdf | SAPLMA: truth probes on hidden states, 71-83% acc |
| 2 | When Truthful Representations Flip Under Deceptive Instructions | 2025 | papers/2507.22149_*.pdf | SAE features flip under deceptive prompts |
| 3 | Can LLMs Lie? Investigation beyond Hallucination | 2025 | papers/2509.03518_*.pdf | Mechanistic analysis of lying, steering vectors |
| 4 | Truth is Universal: Robust Detection of Lies in LLMs | 2024 | papers/2407.12831_*.pdf | Universal truth directions across models |
| 5 | Lies, Damned Lies, and Distributional Language Statistics | 2024 | papers/2412.17128_*.pdf | Review: persuasion & deception in LLMs |
| 6 | Prompt-Induced Linguistic Fingerprints | 2025 | papers/2508.12632_*.pdf | Linguistic signatures in fake news prompts |
| 7 | Probing the Geometry of Truth | 2025 | papers/2506.00823_*.pdf | Linear truth structure in representations |
| 8 | Sleeper Agents | 2024 | papers/2401.05566_*.pdf | Deceptive LLMs persist through safety training |
| 9 | Fake News Detectors Biased against LLM Text | 2023 | papers/2309.12048_*.pdf | Stylistic bias in LLM-generated text |
| 10 | Limitations of Stylometry for Fake News | 2019 | papers/1908.09805_*.pdf | Surface features insufficient for detection |
| 11 | Can LLM-Generated Misinformation Be Detected? | 2023 | papers/2309.13788_*.pdf | Detection challenges for LLM misinformation |
| 12 | Benchmarking Deception Probes | 2025 | papers/2507.12691_*.pdf | Rigorous deception probe evaluation |
| 13 | Building Better Deception Probes | 2026 | papers/2503.17407_*.pdf | Targeted instruction pairs for probes |
| 14 | Caught in the Act | 2025 | papers/2508.19505_*.pdf | Mechanistic deception detection |
| 15 | DeceptionBench | 2025 | papers/2501.13946_*.pdf | Comprehensive deception benchmark |
| + 15 more | See papers/README.md | | | |

See papers/README.md for detailed descriptions.

---

## Datasets
Total datasets downloaded: 5

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| TruthfulQA | HuggingFace | 817 questions | QA truthfulness | datasets/truthfulqa/ | Primary benchmark |
| CounterFact | HuggingFace | 21,919 statements | Factual probing | datasets/counterfact/ | True/false completions |
| FEVER-NLI | HuggingFace | 248K claims | Fact verification | datasets/fever_nli/ | Large-scale verification |
| LIAR | HuggingFace | 12,836 statements | 6-class classification | datasets/liar/ | Graded truthfulness |
| Azaria True-False | Web download | 6,084 statements | Binary true/false | datasets/azaria_true_false/ | 6 topics, SAPLMA paper |

See datasets/README.md for detailed descriptions and download instructions.

---

## Code Repositories
Total repositories cloned: 6

| Name | URL | Purpose | Location |
|------|-----|---------|----------|
| HidingInTheHiddenStates | github.com/sisinflab/... | SAPLMA reproduction & extension | code/HidingInTheHiddenStates/ |
| saplma-probes | github.com/balevinstein/Probes | Truth probes (supervised + CCS) | code/saplma-probes/ |
| geometry-of-truth | github.com/saprmarks/... | Linear truth directions in LLMs | code/geometry-of-truth/ |
| representation-engineering | github.com/andyzoujm/... | RepE: honesty steering vectors | code/representation-engineering/ |
| truthful-representation-flip | github.com/ivyllll/... | SAE analysis of deceptive instructions | code/truthful-representation-flip/ |
| llm-liar | github.com/llm-liar/... | Mechanistic analysis of LLM lying | code/llm-liar/ |

See code/README.md for detailed descriptions.

---

## Resource Gathering Notes

### Search Strategy
1. Used paper-finder service in diligent mode with query "LLM lying deception detection text distribution"
2. Retrieved 141 papers ranked by relevance (scores 1-3)
3. Selected 30 papers most relevant to our specific hypothesis
4. Downloaded via arxiv API with Semantic Scholar for ID resolution

### Selection Criteria
- Papers directly studying truthful vs deceptive LLM text generation
- Papers analyzing internal representations during lying
- Papers on linguistic/stylistic detection of LLM-generated false text
- Seminal papers on LLM deception capabilities
- Papers providing relevant datasets or benchmarks

### Challenges Encountered
- Some arxiv IDs from Semantic Scholar were incorrect; required multiple lookups
- Several API rate limits during Semantic Scholar queries
- Some papers not available on arxiv (conference-only publications)

### Gaps and Workarounds
- No single dataset provides paired truthful/deceptive LLM outputs for the same prompts — will need to generate this in the experiment phase
- Most existing work focuses on internal representations rather than output text distributions — our work fills this gap
- Limited cross-model comparison studies available

---

## Recommendations for Experiment Design

### Primary Dataset(s)
1. **Custom paired corpus** (to be generated): Same prompts answered truthfully and deceptively by multiple LLMs
2. **TruthfulQA**: Established benchmark, good for validation
3. **CounterFact**: Controlled true/false completions for distributional analysis
4. **Azaria True-False**: Simple controlled statements across diverse topics

### Baseline Methods
1. Random classifier (50%)
2. Perplexity-based detection
3. N-gram / vocabulary overlap analysis
4. BERT-based text classifier
5. LLM self-evaluation (few-shot)

### Evaluation Metrics
1. KL divergence / Jensen-Shannon divergence between output distributions
2. Classification accuracy / F1 for truthful vs deceptive text
3. Effect size (Cohen's d) for distributional features
4. Perplexity comparison
5. Vocabulary diversity metrics (TTR, hapax legomena ratio)

### Code to Adapt/Reuse
1. **geometry-of-truth**: Linear probing methodology and truth direction extraction
2. **representation-engineering**: Steering vector extraction for honesty/lying
3. **truthful-representation-flip**: SAE-based feature analysis framework
4. **HidingInTheHiddenStates**: SAPLMA classifier training pipeline
