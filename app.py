import streamlit as st
import pandas as pd
import os
from difflib import SequenceMatcher
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="LLM Evaluation Framework", layout="wide")
st.title("📊 LLM Evaluation Framework")
st.markdown("Evaluate a small LLM (Flan‑T5) on a fixed Q&A dataset.")

# ---------- Dataset ----------
DATASET = [
    {"input": "What is the capital of France?", "expected": "Paris"},
    {"input": "Who wrote 'Hamlet'?", "expected": "Shakespeare"},
    {"input": "What is 2 + 2?", "expected": "4"},
    {"input": "What is the largest planet?", "expected": "Jupiter"},
    {"input": "What is the boiling point of water?", "expected": "100°C"},
    {"input": "Who is the current US president?", "expected": "Joe Biden"},
    {"input": "What is the currency of Japan?", "expected": "Yen"},
    {"input": "What is the tallest mountain?", "expected": "Everest"},
    {"input": "What is the square root of 9?", "expected": "3"},
    {"input": "What year did WW2 end?", "expected": "1945"},
]

# ---------- Load model (cached) ----------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    model.to("cpu")
    return tokenizer, model

tokenizer, model = load_model()

def get_model_answer(question: str) -> str:
    inputs = tokenizer(question, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to("cpu") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# ---------- Scoring functions ----------
def exact_match(answer, expected):
    return answer.strip().lower() == expected.strip().lower()

def llm_as_judge(answer, expected, threshold=0.6):
    return SequenceMatcher(None, answer.lower(), expected.lower()).ratio() >= threshold

def rule_based(answer, expected):
    return expected.lower() in answer.lower()

def combined_score(answer, expected):
    em = exact_match(answer, expected)
    lj = llm_as_judge(answer, expected)
    rb = rule_based(answer, expected)
    return {"exact_match": em, "llm_judge": lj, "rule_based": rb, "overall": em or lj or rb}

# ---------- Evaluation runner ----------
def run_evaluation():
    results = []
    for item in DATASET:
        answer = get_model_answer(item["input"])
        scores = combined_score(answer, item["expected"])
        results.append({
            "question": item["input"],
            "expected": item["expected"],
            "answer": answer,
            **scores
        })
    df = pd.DataFrame(results)
    df.to_csv("eval_results.csv", index=False)
    return df

def get_regression():
    if os.path.exists("eval_results.csv"):
        prev = pd.read_csv("eval_results.csv")
        return prev["overall"].mean()
    return None

# ---------- Streamlit UI ----------
if st.button("🚀 Run Evaluation"):
    with st.spinner("Running evaluation on 10 questions..."):
        df = run_evaluation()
        pass_rate = df["overall"].mean()
        prev_pass = get_regression()
        st.success("Evaluation complete!")
        
        col1, col2 = st.columns(2)
        col1.metric("Overall Pass Rate", f"{pass_rate:.2%}")
        if prev_pass is not None:
            diff = pass_rate - prev_pass
            col2.metric("Change from previous run", f"{diff:+.2%}", delta_color="normal")
        else:
            col2.metric("Previous run", "First run – no regression data")

        st.subheader("Detailed Results")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", data=csv, file_name="eval_results.csv", mime="text/csv")
else:
    st.info("Click the button above to run the evaluation.")
