# QPP Measure Evaluation Across Datasets

## 📈 Results Overview

| Dataset  | Measure        | Macro-F1 | F1 (Clear) | F1 (Ambiguous) | FPR (Clear) |
|----------|--------------|----------|------------|----------------|-------------|
| **CLAQUA** | Max IDF      | **0.589** | 0.603      | **0.575**      | 39.7%       |
|          | Avg. ICTF     | 0.572    | 0.567      | 0.578          | 43.3%       |
|          | Avg. IDF      | 0.558    | 0.543      | 0.573          | 45.8%       |
| **PACIFIC** | SCS          | **0.549** | 0.887      | **0.211**      | 68.7%       |
|          | Avg. VAR      | 0.517    | 0.896      | 0.139          | 74.4%       |
|          | Clarity       | 0.516    | 0.885      | 0.147          | 73.1%       |
| **SIMMIC** | Query Entropy | **0.484** | 0.896      | **0.073**      | 82.4%       |
|          | Query Length  | 0.480    | 0.850      | 0.109          | 62.4%       |
|          | Avg. ICTF     | 0.447    | 0.803      | 0.090          | 61.6%       |

> **Note:**  
> *FPR (Clear)* = proportion of clear queries incorrectly classified as ambiguous.

---

## Key Observations

- **CLAQUA** shows balanced performance across clear and ambiguous queries.  
- **PACIFIC** and **SIMMIC** exhibit:
  - High performance on *clear queries*
  - Very low performance on *ambiguous queries*
  - High false positive rates  

---

<details>
<summary> Why Some Measures Generalise Well (CLAQUA)</summary>

###  IDF-Based Measures on Balanced Data

1. **Signal-to-noise ratio**  
   With ~46.4% ambiguous queries, term statistics provide strong discriminative signals.  
   Rare entity names often correlate with ambiguity in knowledge QA.

2. **Domain diversity**  
   A vocabulary of ~86K across 10+ domains ensures that IDF captures genuine term specificity rather than artefacts.

3. **Feature alignment with ambiguity**  
   Ambiguity types (e.g., entity polysemy, name collisions) align well with low-IDF signals.

</details>

---

<details>
<summary> Why False Positives Dominate (PACIFIC, SIMMIC)</summary>

### ⚖️ Imbalanced Threshold Calibration

- Severe class imbalance:
  - PACIFIC: **16.1% ambiguous**
  - SIMMIC: **4.2% ambiguous**

- Optimising **macro-F1** leads to:
  - Higher recall for ambiguous queries  
  - Lower precision → increased false positives  

---

### Threshold Optimisation Effect

- The `best_f1` method searches over ~200 thresholds to maximise:

$$
\frac{1}{2} \left(F_1(\text{clear}) + F_1(\text{ambiguous})\right)
$$

- With few positive samples:
  - Optimal thresholds shift lower  
  - More queries are labelled as ambiguous  
  - **False positives increase significantly**

---

### Distribution Shift (SIMMIC)

- Training set: **11.1% ambiguous**
- Test set: **4.2% ambiguous**

➡️ Leads to **threshold miscalibration** and degraded generalisation.

</details>

---

## Summary

-  **Balanced datasets (CLAQUA)** → robust performance with IDF-based measures  
-  **Imbalanced datasets (PACIFIC, SIMMIC)** → high false positives due to threshold bias  
-  Proper calibration is critical for real-world deployment  

---

## Takeaway

> QPP measures are sensitive to **dataset balance**, **feature alignment**, and **threshold calibration**.  
Careful tuning is essential when transferring models across domains.

---
