# 📊 LLM Evaluation Framework

An automated evaluation harness for testing and benchmarking LLMs on a fixed Q&A dataset. Built with **Streamlit** and **Flan‑T5‑small**, it scores model outputs using multiple metrics and tracks performance regressions across runs.

---

## 🚀 Live Demo

**[👉 Click here to run the live demo](https://nuwqyyxev25qc2ssjm7tuk.streamlit.app/)**  


## ✨ Features

- **Automated Test Suite** – Runs 10 fixed Q&A pairs through the model.
- **Multi‑Metric Scoring** – Evaluates outputs with:
  - ✅ **Exact Match** – Strict string equality.
  - ✅ **LLM-as-Judge** – Text similarity using `SequenceMatcher` (threshold 0.6).
  - ✅ **Rule‑Based** – Checks if the expected answer appears in the output.
  - ✅ **Overall** – Passes if *any* of the three metrics pass.
- **Regression Tracking** – Saves results to CSV and compares pass rate against the previous run.
- **Exportable Results** – Download the full evaluation table as a CSV file.
- **Clean Dashboard** – Built with Streamlit for easy visualisation.

---

## 🏗️ Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend / UI | [Streamlit](https://streamlit.io/) |
| LLM (evaluated) | [Flan‑T5‑small](https://huggingface.co/google/flan-t5-small) |
| Scoring | Python (`difflib` for similarity) |
| Deployment | [Streamlit Cloud](https://streamlit.io/cloud) |
| Language | Python 3.9+ |

---

## 📊 Sample Results

The framework correctly identified **1 out of 10** questions (10% pass rate) for Flan‑T5‑small. This low accuracy is expected for a tiny model – **the value is the framework itself**, not the score.

| Question | Expected | Model Answer | Pass |
|----------|----------|--------------|------|
| What is the capital of France? | Paris | l'ondre | ❌ |
| Who wrote 'Hamlet'? | Shakespeare | john scott | ❌ |
| What is 2 + 2? | 4 | a slender slender | ❌ |
| What is the largest planet? | Jupiter | venus | ❌ |
| What is the boiling point of water? | 100°C | a vapor | ❌ |
| Who is the current US president? | Joe Biden | john w. bush | ❌ |
| **What is the currency of Japan?** | **Yen** | **yen** | **✅** |
| What is the tallest mountain? | Everest | st johns | ❌ |
| What is the square root of 9? | 3 | (empty) | ❌ |
| What year did WW2 end? | 1945 | 1912 | ❌ |

📄 **Full results CSV:** [`eval_results.csv`](eval_results.csv)

---

## 📁 How to Run Locally

1. **Clone the repository**  
   ```bash
   git clone https://github.com/Dhayalramesh/-LLM-Evaluation-Framework.git
   cd -LLM-Evaluation-Framework
2.pip install -r requirements.txt
3.streamlit run app.py
