<h2 align="center">🔬 QPP for (🤖Conversational) ASSISTANT</h2>

![Sci-Fi Banner Image](https://github.com/importrayhan/QPP_4_ASSISTANT/blob/main/convir_qpp.png?raw=true)
# QPP Heuristic Baseline

Query Performance Prediction (QPP) as an unsupervised baseline for scoring
conversational search system effectiveness followed by ambiguous turn classification.

**Hypothesis**: ambiguous queries have lower QPP scores because they are
harder to satisfy with retrieved information. The retrieval signal is
weaker, term specificity is lower, and the retrieved documents are less
coherent.

<div align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Roboto&size=29&duration=3000&pause=800&color=F7F7F7&background=003F7BBD&multiline=true&width=1028&height=480&lines=%3C%7C+Relevant+Document+%7C%3E;User%3A+What+percentage+of+total+non-marketable+investments+were+that+in+2019%3F;Query+Entropy+(QPP)+%3A+9.8817;Prediction_label%3A+%22clear%22;%22Ground_truth%22%3A+%22clear%22;User%3A+What+was+the+total+accumulated+amortization%3F;Query+Entropy+(QPP)+%3A+3.9691;Prediction_label%3A+%22ambiguous%22;%22Ground_truth%22%3A+%22ambiguous%22" alt="Typing SVG" />
</div>


## Usage

### Basic evaluation 

```bash
python scripts/qpp_evaluate.py \
    --benchmark pacific \
    --num_classes 2 \
    --score_key avg_idf \
    --threshold_method best_f1
```

This computes pre-retrieval QPP measures for every system turn, finds the
optimal threshold on the train set, applies it to the test set, and
reports the same metrics as the BiLSTM-CRF baseline.

### With mock ranked list (enables POST-RETRIEVAL)

```bash
python scripts/qpp_evaluate.py \
    --benchmark pacific \
    --mock_ranked_list \
    --n_irrelevant 10 \
    --score_key nqc \
    --threshold_method best_f1
```

For each turn, builds a ranked list by BM25-scoring the turn's observations
(relevant) plus 10 random observations from other conversations
(irrelevant).  This enables WIG, NQC, SMV, sigma_max, n_sigma, clarity.

### With T5 query rewriter

```bash
python scripts/qpp_evaluate.py \
    --benchmark pacific \
    --rewriter_path /models/t5-query-rewriter \
    --score_key scs
```

Rewrites the query using a local T5 model before computing QPP features.
## Example: per_turn = false:
### Dataset: CLaQuA (last-turn only)

```bash
python scripts/qpp_evaluate.py \
    --benchmark claqua \
    --per_turn false \
    --score_key scs \
    --threshold_method otsu
```

## Example: Retrieval Augmented Generation (RAG) Conversation Assistant:
### With pyserini index (if available)

```bash
python scripts/qpp_evaluate.py \
    --benchmark pacific \
    --index_path /path/to/lucene/index \
    --score_key nqc
```
---

## Output format

Results are saved as JSON for comprehensive evaluation format:

```json
{
  "benchmark": "pacific",
  "score_key": "avg_idf",
  "threshold_method": "best_f1",
  "threshold_used": 3.14,
  "metrics": {
    "accuracy": 0.72,
    "precision": 0.68,
    "recall": 0.71,
    "f1": 0.69,
    "auc_roc": 0.74
  },
  "per_measure_sweep": {
    "avg_idf":  {"f1_weighted": 0.69, "f1_macro": 0.58, ...},
    "scs":      {"f1_weighted": 0.71, "f1_macro": 0.60, ...},
    "nqc":      {"f1_weighted": 0.65, "f1_macro": 0.55, ...}
  }
}
```

Per-turn diagnostics are saved separately with all QPP features:

```json
{
  "turn_index": 3,
  "query": "What is the interest rate for...",
  "prediction": 1,
  "prediction_label": "ambiguous",
  "ground_truth": 1,
  "ground_truth_label": "ambiguous",
  "correct": true,
  "qpp_score": 2.31,
  "qpp_features": {
    "query_length": 6.0,
    "avg_idf": 2.31,
    "scs": 4.52,
    "wig": 1.23,
    "nqc": 0.87
  }
}
```
![Terminal Screenshot](https://github.com/importrayhan/QPP_4_ASSISTANT/blob/main/scripts/terminal.png?raw=true)

---

## Function reference

| Function / Class | File | Purpose |
|---|---|---|
| `PseudoCollection` | qpp_measures.py | Indexes observation texts. Provides `idf(t)`, `collection_prob(t)`, `scq(t)`, `bm25_score()`, `get_random_doc()`. |
| `QPPScorer` | qpp_measures.py | Computes all QPP features for a query. `score_turn()` returns a dict of all applicable measures. |
| `parse_sip_for_qpp()` | qpp_measures.py | Parses SIP conversation into per-turn records with query, observations, and label separated. |
| `threshold_classify()` | qpp_measures.py | Converts QPP score array to binary predictions via percentile, fixed, or Otsu thresholding. |
| `find_best_threshold()` | qpp_measures.py | Grid-searches for the threshold maximizing macro-F1 on labeled data. |
| `qpp_evaluate.py` | scripts/ | End-to-end evaluation script. Builds collection, scores all turns, thresholds, evaluates, sweeps all measures. |

---
## How the QPP engine works (specially when a collection index is not available, it generates a pseudo-collection from conversation history)
![Workflow](https://github.com/importrayhan/QPP_4_ASSISTANT/blob/main/scripts/categories.png?raw=true)

---

## Mathematical formulas

### (1) — Query text only

**Query Length**: `QL(Q) = |{t ∈ Q : t ∉ stopwords}|`

Longer, more specific queries tend to be less ambiguous.

**Query Entropy**: `H(Q) = −Σ_{t∈Q} P(t|Q) · log₂ P(t|Q)`

where `P(t|Q) = tf(t,Q) / |Q|`.  Higher entropy means more diverse
vocabulary within the query.

### (2) — Query + pseudo-collection

The pseudo-collection is built from all observation texts in the dataset.
Each observation is one document.  N = total documents, df_t = documents
containing term t, cf_t = total occurrences of term t.

**AvgIDF**: `AvgIDF(Q) = (1/|Q|) · Σ_{t∈Q} ln(1 + N/df_t)`

Average inverse document frequency.  High values mean query terms are
rare in the collection (more specific, less ambiguous).

**MaxIDF**: `MaxIDF(Q) = max_{t∈Q} ln(1 + N/df_t)`

The most specific term in the query.

**SCQ (Simplified Collection-Query similarity)**:

```
SCQ(t) = (1 + ln(cf_t)) · ln(1 + N/df_t)
AvgSCQ(Q) = (1/|Q|) · Σ SCQ(t)
MaxSCQ(Q) = max SCQ(t)
SumSCQ(Q) = Σ SCQ(t)
```

Combines collection frequency with IDF.  Terms that are frequent overall
but concentrated in few documents score high.

**SCS (Simplified Clarity Score)**:

```
SCS(Q) = Σ_{t∈Q} P_ml(t|Q) · log₂(P_ml(t|Q) / P(t|C))
```

KL-divergence between query language model and collection language model.
High SCS means the query is focused on specific topics that differ from
the collection average — indicating a clear, unambiguous query.

**Query Scope**: `ω(Q) = −log(n_Q / N)`

where n_Q = documents containing at least one query term.  High values
mean fewer documents match (more specific query).

### (3) — Post-retrieval (ranked list required)

These methods operate on a ranked list of (document, score) pairs.
When no pyserini index is available, a mock ranked list is built by
scoring observations (relevant docs) and random observations from other
conversations (irrelevant docs) with BM25.

**WIG**: `WIG(q) = (1/k·√|q|) · Σ_{i=1..k} (s_i − μ_corpus)`

How much the top-k documents outscore the corpus average.

**NQC**: `NQC(q) = std(top_k_scores) / μ_corpus`

Standard deviation of top-k scores, normalized.  High variance means
clear separation between relevant and non-relevant.

**SMV**: `SMV(q) = [Σ s_i · |ln(s_i/μ)|] / (k · μ_corpus)`

Combines magnitude and variance of retrieval scores.

**σ_max**: `σ_max(q) = max_{k'=1..K} std(s_1..s_{k'})`

Maximum standard deviation over all rank prefixes.  Self-selects the
optimal cutoff without a fixed k parameter.

**n(σ_x%)**: std of scores above x% of the max score, normalized by √|q|.

**Clarity Score**: Full KL-divergence between relevance model (built from
top-k documents via Dirichlet smoothing) and collection model.

---

## Thresholding methods

| Method | Description |
|---|---|
| `percentile` | Bottom X% of scores classified as ambiguous (default X=25) |
| `otsu` | Automatic threshold maximizing inter-class variance |
| `best_f1` | Grid search on train set for threshold maximizing macro-F1 |
| `fixed` | Manual threshold value |

For a fair comparison with BiLSTM-CRF, use `best_f1` which tunes on
the train set (not test).

---

## References

- Cronen-Townsend et al., "Predicting Query Performance", SIGIR 2002 (Clarity Score)
- He & Ounis, "Inferring Query Performance Using Pre-retrieval Predictors", SPIRE 2004 (AvgIDF, SCQ)
- Zhou & Croft, "Query Performance Prediction in Web Search Environments", SIGIR 2007 (WIG)
- Shtok et al., "Predicting Query Performance by Query-Drift Estimation", TOIS 2012 (NQC)
- Cummins, "Improved Query Performance Prediction Using Standard Deviation", SIGIR 2014 (σ_max)
- Meng et al., "Query Performance Prediction: From Ad-hoc to Conversational Search", SIGIR 2023 (QPP4CS)
