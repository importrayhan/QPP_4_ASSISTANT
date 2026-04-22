## How each measure estimates query performance

###  Pre-retrieval measures (no ranked list needed)

These measures estimate query difficulty from the query text and
corpus-level term statistics.  They do NOT look at any retrieved
documents — they predict how well retrieval WILL perform.

**AvgIDF** (He & Ounis, ECIR 2008)

```
AvgIDF(Q) = (1/|Q|) · Σ_{t∈Q} log₂(N / df(t))
```

Rationale: IDF measures term specificity.  A query whose terms appear in
few documents has high IDF — the terms are discriminative.  The retrieval
system can easily identify relevant documents because the query terms are
rare and therefore informative.  An ambiguous query tends to use common
terms (low IDF) that match many documents equally, making it hard for the
system to distinguish relevant from irrelevant.

Reference: Our implementation uses `log₂(N/df)` matching the original paper. 

**MaxIDF** — same formula, takes max instead of mean.  Captures the single
most discriminative term.  More robust to queries mixing one rare term with
several common ones.

**AvgICTF** (inverse collection term frequency)

```
AvgICTF(Q) = (1/|Q|) · Σ_{t∈Q} log₂(|C| / cf(t))
```

Like IDF but uses total term occurrences rather than document counts.
A term appearing 1000 times in 100 documents has the same IDF as one
appearing 100 times in 100 documents, but lower ICTF.  ICTF penalizes
terms that are repeated heavily even within few documents.

**SCQ** (Simplified Collection-Query similarity, He & Ounis 2008)

```
SCQ(t) = (1 + ln(cf(t))) · log₂(N / df(t))
```

Multiplies a frequency signal `(1 + ln cf)` by a specificity signal
`idf`.  High SCQ means a term is both frequent enough to be meaningful
AND specific enough to be discriminative.  Three aggregations (Avg, Max,
Sum) capture different aspects: AvgSCQ is the mean signal per term,
MaxSCQ is the peak signal, SumSCQ rewards queries with many informative
terms.

**VAR** (term frequency variance, He & Ounis 2008)

```
VAR(t) ≈ (cf(t)/N) · (1 - df(t)/N)
```

Approximation of the variance of tf(t) across all N documents.  When a
term is concentrated in a few documents (high tf in those, zero elsewhere),
VAR is large — the term is a strong discriminator.  Uniform terms (appear
once in every doc) have low VAR and low discriminative power.

Note: the exact formula requires per-document term frequencies.  Without a
full positional index, we use the Bernoulli approximation, which is the
standard approach when per-doc tf is unavailable (used in multiple QPP
implementations that work from aggregate statistics).

**SCS** (Simplified Clarity Score, He & Ounis 2008)

```
SCS(Q) = Σ_{t∈Q} P_ml(t|Q) · log₂( P_ml(t|Q) / P(t|C) )
```

KL-divergence between the query language model and the collection language
model.  If the query "sounds different" from the collection (high SCS),
it is focused on a specific topic — the retrieval system can find
matching documents easily.  If the query "sounds like the collection"
(low SCS), it is generic and ambiguous.

Critical: SCS is computed over ALL query tokens including stopwords.  This
is because the divergence measures overall language distribution, not just
content terms.  A query full of stopwords has SCS ≈ 0 because stopwords
have similar probabilities in both query and collection.

For OOV terms (query terms not in collection), we use Laplace smoothing
`P(t|C) = 1/|C|` to avoid log(0).

**Query Scope** (He & Ounis 2008)

```
ω(Q) = −log₂(n_Q / N)
```

n_Q = number of documents containing at least one query term.  High query
scope means few documents match → more specific query.  Unlike IDF which
is per-term, query scope considers the combined selectivity of all terms
together.

**Query Length** — simply |Q| after stopword removal.  Longer queries are
generally more specific.  Extremely short queries (1-2 terms) are more
likely to be ambiguous.

**Query Entropy** — Shannon entropy of the term distribution within the
query.  Queries that repeat one term have low entropy (focused); queries
with diverse vocabulary have high entropy.  In conversational settings
this is less informative because most queries are short and have unique
terms.

### Post-retrieval measures (ranked list required)

These measures analyze the score distribution of retrieved documents.
When no search index is available, we build a mock ranked list by
BM25-scoring the turn's observations (as relevant documents) plus random
observations from other conversations (as irrelevant documents).

**WIG** (Zhou & Croft, SIGIR 2007)

```
WIG(Q) = (1/k) · Σ_{i=1..k} (s_i − μ_C) / √|Q|
```

μ_C is the corpus mean score (mean over all scored documents, not just
top-k).  WIG measures how much the top-k documents outscore the average.
If the top documents score much higher than average, the query is "easy"
(clear separation).  If top scores are barely above average, the query is
hard (ambiguous).

The `1/√|Q|` normalization accounts for longer queries having higher
absolute scores due to more matching terms.

Note: k=5 is the canonical default.  QPP4CS uses k∈{5,10,15,20,25,50,100}
as a hyperparameter.

**NQC** (Shtok et al., TOIS 2012)

```
NQC(Q) = σ(s_1, ..., s_k) / μ_C
```

Standard deviation of top-k retrieval scores, normalized by the corpus
mean.  The intuition: when the top documents have diverse scores (some
very high, some moderate), the retrieval system has found a clear relevance
gradient.  When all top scores are similar (low σ), the system cannot
distinguish relevant from irrelevant — the query is ambiguous.

Note: k=100 is the canonical default.  σ is the sample standard deviation.
The division by μ_C controls for queries that produce uniformly higher or
lower scores due to length or collection effects.

**SMV** (Tao & Wu, CIKM 2014)

```
SMV(Q) = (1/(k·μ_C)) · Σ_{i=1..k} s_i · |ln(s_i / μ_k)|
```

Combines magnitude (s_i) with deviation from the top-k mean (|ln(s_i/μ_k)|).
Documents that both score highly AND differ from the top-k mean contribute
most.  This avoids a failure mode of NQC where variance can be high due to
a single outlier.

**σ_max** (Cummins, SIGIR 2014)

```
σ_max(Q) = max_{k'=2..K} σ(s_1, ..., s_{k'})
```

Computes σ at every rank prefix and takes the maximum.  This self-selects
the rank cutoff: some queries have clear score drops at rank 5, others at
rank 50.  σ_max adapts automatically.  It was the most robust predictor
in QPP4CS across conversational search settings.

**n(σ_x%)** (Pérez-Iglesias & Araujo 2010)

```
n(σ_{x%})(Q) = σ({s_i : s_i ≥ x% · s_max}) / √|Q|
```

Filters to only high-scoring documents (above x% of the top score), then
computes σ.  This focuses on the "relevant cluster" and ignores the noise
from clearly irrelevant documents.  Default x=50%.

**Clarity Score** (Cronen-Townsend et al., SIGIR 2002)

```
Clarity(Q) = Σ_w P(w|θ_Q) · log₂( P(w|θ_Q) / P(w|C) )
```

Full KL-divergence between the relevance model (estimated from top-k
documents) and the collection model.  The relevance model is built using
Dirichlet-smoothed document language models:

```
P(w|θ_Q) = Σ_d P(d|Q) · P_dir(w|d)
P(d|Q) = Score(Q,d) / Σ Score(Q,d')
P_dir(w|d) = (tf(w,d) + μ·P(w|C)) / (|d| + μ)
```

μ=2000 is the standard Dirichlet smoothing parameter.  Clarity was
explicitly designed to "quantify query ambiguity."  If the top-ranked
documents converge on a single topic (high clarity), the query is
unambiguous.  If they span diverse topics (low clarity), the query is
ambiguous.

---

## How thresholds are set

The QPP score is a continuous value.  To classify binary ambiguity, we
need a decision boundary.  Four methods are available:

**`best_f1`** (recommended for benchmarking): Grid-searches 200 thresholds
between min and max QPP scores on the training set, evaluates macro-F1 at
each threshold, and selects the one maximizing macro-F1.  This threshold
is then applied to the test set.  This is the fairest comparison with
BiLSTM-CRF because both use training data to learn their decision
boundary.  The risk: overfitting to training distribution if the training
set is small.

**`otsu`** (recommended for truly unsupervised use): Otsu's method finds
the threshold that maximizes between-class variance:

```
t* = argmax_t  w₀(t) · w₁(t) · (μ₀(t) − μ₁(t))²
```

where w₀, w₁ are the fraction of samples below/above t, and μ₀, μ₁ are
their means.  This assumes the score distribution is bimodal (one cluster
of easy queries, one of hard queries).  It uses NO labels — purely
unsupervised.  However, if the true classes have overlapping score
distributions (which is typical), Otsu may find a suboptimal split.

**`percentile`**: Labels the bottom X% as ambiguous.  Default X=25%
(matching the typical 12-25% ambiguity rate in conversational datasets).
This requires knowing the approximate class ratio — not truly
unsupervised but avoids the bimodality assumption of Otsu.

**`fixed`**: Manual threshold.  Useful when the threshold has been
calibrated on a development set or when applying a known boundary (e.g.,
the 1.27-bit Clarity threshold from Cronen-Townsend et al.).

---

## How per_turn processes the input

When `--per_turn true` (pacific, simmic):

```
Conversation: [u₀, obs₀, gpt₀, u₁, obs₁, gpt₁, u₂, obs₂, gpt₂]

Turn 0: query=u₀  observations=[obs₀]  label=gpt₀.ambiguous_type
Turn 1: query=u₁  observations=[obs₁]  label=gpt₁.ambiguous_type
Turn 2: query=u₂  observations=[obs₂]  label=gpt₂.ambiguous_type

→ 3 QPP scores, 3 predictions, 3 labels compared
```

Even when a query rewriter path is provided, each rewritten turn is scored independently.  The mock ranked list for turn k uses
obs_k as relevant documents and random observations from other
conversations as irrelevant documents.  Context queries [u₀, ..., u_{k-1}]
are used for the query_drift feature.

When `--per_turn false` (claqua):

```
Same conversation, but only the LAST turn is scored:

Turn 2: query=u₂  observations=[obs₂]  label=gpt₂.ambiguous_type

→ 1 QPP score, 1 prediction, 1 label compared
```

Prior turns are ignored for QPP scoring.  The QPP baseline is
fundamentally per-query, so per_turn=false simply restricts which queries
are evaluated.

---

## False positives

False positives occur when the model predicts "ambiguous" but the ground
truth is "clear."  In QPP, this means the QPP score is below the threshold
for a query that is actually well-served by the retrieval system.

### Mitigation: 
`percentile` method calibrated to the known class ratio.

### SCS with stopwords inflates scores for vague queries

The canonical SCS includes stopwords in the KL-divergence computation.
On domain-specific pseudo-collections (e.g. finance observations),
stopwords like "tell", "me", "about" have very low P(t|C) because the
collection is full of domain terms.  A vague query like "tell me about
money" gets HIGH SCS because its stopwords diverge strongly from the
collection language — the opposite of what we want.

Fix implemented: `scs_content` excludes stopwords from the computation.
This variant correctly scores specific queries higher than vague ones.
Use `--score_key scs_content` instead of `scs` for ambiguity detection.

### OOV terms produce misleading extremes

A query whose content terms are ALL out-of-vocabulary (e.g. "tell me
about money" where "money" never appears in any observation) gets:
- AvgIDF = 0 (correct: no collection signal)
- SCS (canonical) = very high (wrong: OOV terms get Laplace-smoothed
  maximum divergence)
- SCS_content = 0 (correct: guard returns 0 when all terms are OOV)

For IDF-based measures, OOV → 0 is conservative (assumes no signal).
For SCS, OOV inflates divergence.  Our `scs_content` guards against
this by returning 0 when no content terms have collection presence.

The BERT tokenizer option (`--bert_tokenizer_path`) reduces OOV rates
because WordPiece always produces in-vocabulary subwords, but the
subwords may not carry meaningful IDF.

### Pseudo-collection mismatch

The pseudo-collection is built from observation texts, which are
post-retrieval results.  If observations are short snippets rather than
full documents, the collection statistics (df, cf) are noisy.  A term
that appears in many observations may have artificially high df,
deflating its IDF.  Conversely, terms from a specific domain may be
rare in observations but common in the actual document collection.

Mitigation: use --mock_ranked_list to enable post-retrieval measures (WIG, NQC,
sigma_max) which measure score distributions rather than raw term
statistics.  These are more robust to collection mismatch because they
operate on relative scores.

### Multi-word concepts tokenized as separate terms

"New York" becomes ["new", "york"].  "New" has low IDF (very common),
dragging down AvgIDF.  The query is actually specific (about New York)
but the per-term aggregation misses the bigram specificity.

Mitigation: MaxIDF is more robust here (captures "york" which is rare).
SumSCQ also handles this because it accumulates signal across terms
rather than averaging.  The BERT tokenizer partially addresses this
through subword semantics.

### Context-dependent queries in multi-turn conversations

"What about the other one?" is highly ambiguous in isolation but
perfectly clear in conversational context.  Pre-retrieval QPP measures
see only the current query text and have no access to conversational
history.

Mitigation: the query_drift feature captures how much the current query
differs from the first query, but this is a weak signal.  A query
rewriter (--rewriter_path) can resolve coreferences before QPP scoring,
transforming "the other one" into "the other savings account with higher
interest rate" which has much higher IDF.

### Mock ranked list quality

When using --mock_ranked_list, the irrelevant documents are random
observations from other conversations.  If these random observations
happen to be topically similar to the current query, the BM25 scores
of irrelevant documents will be high, making the score distribution
look uniform (low NQC/WIG) even for a clear query.

Mitigation: increase --n_irrelevant (default 10) to 50+ for more
stable estimates.  With more random documents, the probability that
many are topically similar decreases.

---

## Formula validation summary

| Measure | Paper | Log base |  Validated against |
|---------|-------|----------|-------------------|
| AvgIDF  | He & Ounis 2008 | log₂ | [Narabzad pre_retrievals.py](https://github.com/Narabzad/Query-Performance-Prediction/blob/master/Pre-retrieval/code/pre_retrievals.py) |
| MaxIDF  | He & Ounis 2008 | log₂ | [Narabzad pre_retrievals.py](https://github.com/Narabzad/Query-Performance-Prediction/blob/master/Pre-retrieval/code/pre_retrievals.py) |
| AvgICTF | He & Ounis 2008 | log₂ | [Narabzad pre_retrievals.py](https://github.com/Narabzad/Query-Performance-Prediction/blob/master/Pre-retrieval/code/pre_retrievals.py) |
| SCQ     | He & Ounis 2008 | ln(cf)·log₂(idf) |  [Narabzad pre_retrievals.py](https://github.com/Narabzad/Query-Performance-Prediction/blob/master/Pre-retrieval/code/pre_retrievals.py) |
| SCS     | He & Ounis 2008 | log₂ | [QPP4CS](https://github.com/ChuanMeng/QPP4CS/blob/main/unsupervisedQPP/pre_retrieval.py) |
| ω(Q)    | He & Ounis 2008 | log₂ | — |
| WIG     | Zhou & Croft 2007 | — | [QPP4CS](https://github.com/ChuanMeng/QPP4CS) |
| NQC     | Shtok et al. 2012 | — | [QPP4CS](https://github.com/ChuanMeng/QPP4CS), [QPPTK](https://github.com/Zendelo/QPP-EnhancedEval/tree/main/code/python) |
| SMV     | Tao & Wu 2014 | ln | [QPP4CS](https://github.com/ChuanMeng/QPP4CS) |
| σ_max   | Cummins 2014 | — | [QPP4CS](https://github.com/ChuanMeng/QPP4CS) |
| Clarity | Cronen-Townsend 2002 | log₂ | [QPP4CS](https://github.com/ChuanMeng/QPP4CS)|
| BM25    | Robertson et al. | — | pyserini (k1=0.9, b=0.4) |

---

## References

1. He, B. & Ounis, I. (2008). Using Coherence-Based Measures to Predict
   Query Difficulty. ECIR.
2. Zhou, Y. & Croft, W.B. (2007). Query Performance Prediction in Web
   Search Environments. SIGIR.
3. Shtok, A. et al. (2012). Predicting Query Performance by Query-Drift
   Estimation. TOIS.
4. Tao, Y. & Wu, S. (2014). Query Performance Prediction by Considering
   Score Magnitude and Variance Together. CIKM.
5. Cummins, R. (2014). Improved Query Performance Prediction Using Standard
   Deviation. SIGIR.
6. Cronen-Townsend, S. et al. (2002). Predicting Query Performance. SIGIR.
7. Pérez-Iglesias, J. & Araujo, L. (2010). Standard Deviation as a Query
   Hardness Estimator. SPIRE.
8. Meng, C. et al. (2023). Query Performance Prediction: From Ad-hoc to
   Conversational Search. SIGIR.
9. Mo, F. et al. (2026). OpenDecoder: Open LLM Decoding to Incorporate
   Document Quality in RAG. arXiv:2601.09028.
