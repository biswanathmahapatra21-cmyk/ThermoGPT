"""
backend/app_streamlit.py
ThermalGPT — Streamlit Mini POC
Includes:
- Semantic Retrieval (PDF corpus)
- OpenAI Expert Reasoning
- Groq Llama-3 Expert Reasoning (free)
- Calculation environment
"""

# ----------------------------------------------------------
# Path fix (Windows safe import)
# ----------------------------------------------------------
import sys, os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ----------------------------------------------------------
# Imports
# ----------------------------------------------------------
import streamlit as st
from backend.retriever import Retriever
from backend.executor import run_calculation

from dotenv import load_dotenv
load_dotenv()

import os
from openai import OpenAI          # New OpenAI API
from groq import Groq              # Groq API for free models

# Load keys
openai_key = os.getenv("OPENAI_API_KEY")
groq_key = os.getenv("GROQ_API_KEY")

# Create clients
openai_client = OpenAI(api_key=openai_key) if openai_key else None
groq_client = Groq(api_key=groq_key) if groq_key else None


# ----------------------------------------------------------
# LLM Functions
# ----------------------------------------------------------
def generate_answer_openai(query, retrieved_docs):
    context = "\n\n".join(retrieved_docs)

    prompt = f"""
You are a senior thermal engineering expert.
Use the following context to answer the user's question clearly and precisely.

Context:
{context}

Question: {query}

Provide derivations, formulas, and short Python examples if relevant.
"""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert thermal engineer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message["content"]


def generate_answer_groq(query, retrieved_docs):
    context = "\n\n".join(retrieved_docs)

    prompt = f"""
You are a senior thermal engineering expert.
Use the following context to answer the user's question clearly and precisely.

Context:
{context}

Question: {query}

Provide derivations, formulas, and short Python examples if relevant.
"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are an expert thermal engineer."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2
    )
    return response.choices[0].message.content



# ----------------------------------------------------------
# UI Setup
# ----------------------------------------------------------
st.set_page_config(page_title="ThermalGPT — Mini POC", layout="wide")
st.title("🔥 ThermalGPT — Mini Proof of Concept")

st.markdown("""
Welcome to **ThermalGPT**, your domain expert for  
**Thermal Engineering** and **Fluid Mechanics**.

💡 Ask theory questions, derivations, or perform live calculations.
""")

# Initialize retriever
retriever = Retriever()

if retriever.nn is None:
    st.error("❌ No index loaded. Please run `python backend/ingest_corpus.py` after adding PDFs into `data/corpus`.")
    st.stop()


# ----------------------------------------------------------
# Sidebar
# ----------------------------------------------------------
with st.sidebar:
    st.header("LLM Provider Selection")
    model_choice = st.radio(
        "Choose Expert Reasoning Model:",
        ["OpenAI GPT-4o-mini", "llama-3.3-70b-versatile"]
    )

    st.divider()
    st.markdown("📁 Corpus Loaded")
    st.success(f"Chunks Loaded: {len(retriever.texts)}")


# ----------------------------------------------------------
# User Query
# ----------------------------------------------------------
query = st.text_area("Enter your Thermal/Fluid Mechanics question:", height=140)

if st.button("Get Expert Answer"):
    if not query.strip():
        st.warning("⚠️ Please enter a question.")
        st.stop()

    # Retrieve relevant text
    with st.spinner("🔍 Retrieving relevant domain knowledge..."):
        docs = retriever.retrieve(query, k=4)

    st.subheader("📘 Retrieved Context")
    for d in docs:
        st.markdown(f"- {d}")

    st.markdown("---")

    # Generate expert reasoning
    st.subheader("🧠 Expert Reasoning")
    with st.spinner("Generating expert response..."):
        try:
            if model_choice == "OpenAI GPT-4o-mini":
                answer = generate_answer_openai(query, docs)
            else:
                answer = generate_answer_groq(query, docs)

            st.markdown(answer)

        except Exception as e:
            st.error(f"⚠️ Error generating answer: {e}")

    st.markdown("---")

    # Auto-show calculator for numeric requests
    if any(word in query.lower() for word in ["calculate", "compute", "heat", "capacity", "power", "flow"]):
        st.subheader("🧮 Quick Calculation Tool")
        m_dot = st.number_input("Mass flow rate (kg/s)", value=1.0)
        cp = st.number_input("Specific heat (J/kg·K)", value=4186.0)
        t_in = st.number_input("Inlet temperature (°C)", value=20.0)
        t_out = st.number_input("Outlet temperature (°C)", value=60.0)

        if st.button("Run Calculation"):
            code = f"result = {{'Q': {m_dot} * {cp} * ({t_out} - {t_in})}}\nprint(json.dumps(result))"
            out, err = run_calculation(code)
            if err:
                st.error(err)
            else:
                st.success("Heat Transfer Result (Watts):")
                st.code(out)


st.caption("Built with ❤️ using Streamlit + OpenAI + Groq · © 2025 ThermalGPT")


