import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000/query"

st.set_page_config(page_title="Agentic PDF Assistant", layout="wide")

st.title("📄 Agentic PDF Assistant")

st.write(
    "Upload a PDF and ask anything. "
    "Your query is processed by an AI agent running on the backend."
)

# ---------------- SESSION STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------- Upload PDF --------------------
uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

# -------------------- FORM (IMPORTANT) --------------------
with st.form("agent_form", clear_on_submit=True):

    user_query = st.text_area(
        "Ask your question",
        height=120,
        placeholder=(
            "Ask anything about the document...\n"
            "Example:\n"
            "- Extract the resume\n"
            "- Generate MCQs\n"
            "- Check skill gaps"
        )
    )

    submitted = st.form_submit_button("Run Agent")

# -------------------- Run Agent --------------------
if submitted:

    if not uploaded_file:
        st.error("Please upload a PDF")
        st.stop()

    if not user_query.strip():
        st.error("Please enter a question")
        st.stop()

    with st.spinner("Agent is thinking..."):

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                "application/pdf"
            )
        }

        data = {"question": user_query}

        try:
            response = requests.post(
                BACKEND_URL,
                files=files,
                data=data,
                timeout=300
            )

            if response.status_code != 200:
                st.error(response.json().get("error", "Backend error"))
            else:
                st.session_state.history.append({
                    "question": user_query,
                    "answer": response.json()["answer"]
                })

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to backend. Is Flask running?")
        except Exception as e:
            st.error(str(e))

# -------------------- DISPLAY HISTORY --------------------
if st.session_state.history:
    st.subheader("🧠 Conversation History")

    for idx, item in enumerate(st.session_state.history, start=1):
        st.markdown(f"### {idx}")
        st.markdown("**Question:**")
        st.write(item["question"])
        st.markdown("**Answer:**")
        st.write(item["answer"])
        st.divider()
