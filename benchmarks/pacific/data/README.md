# 🌊 PACIFIC Dataset  
**ProActive ConversatIonal Question Answering in FInanCe**

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2210.08817)
[![Dataset](https://img.shields.io/badge/Dataset-GitHub-blue)](https://github.com/dengyang17/PACIFIC/)
[![Conference](https://img.shields.io/badge/EMNLP-2022-green)](#)
[![Domain](https://img.shields.io/badge/Domain-Finance-orange)](#)
[![Task](https://img.shields.io/badge/Task-Proactive%20QA-purple)](#)

---

## 📊 Overview

| Dataset  | Domain  | Turns | Modality       | Proactive | Numerical Reasoning |
|----------|---------|-------|----------------|-----------|---------------------|
| PACIFIC  | Finance | Multi | Table / Text   | ✅ Yes    | ✅ Yes              |

---

### 🔁 We transform the PACIFIC dataset into the following structure for use in our QPP query understanding engine.

```json
List of Conversations [
  {
    "conversation": [
      {
        "from": "human | function_call | observation | gpt",
        "value": "...",
        "ambiguous_type": "0 | 1 | 2(opt.) | 3(opt.)",
        "metadata": { ... }
      }
    ],
    "metadata": { ... }
  }
]

