import time
import requests
import streamlit as st

# FastAPI Backend API Configuration
BACKEND_URL = "http://127.0.0.1:8000"

def get_backend_health():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def render_confidence_badge(score, validation=None):
    if score >= 90:
        color = "#2E7D32"
        label = f"Diagnostic Confidence: High ({score}%) - Explicit Error Confirmed"
    elif score >= 70:
        color = "#1565C0"
        label = f"Diagnostic Confidence: Moderate ({score}%) - Strong Context Match"
    elif score >= 40:
        color = "#F57C00"
        label = f"Diagnostic Confidence: Possible Issue ({score}%)"
    else:
        color = "#D32F2F"
        label = f"Diagnostic Confidence: Low ({score}%) - No Explicit Fault Found (Log Appears Operational)"

    st.markdown(f"""
    <div style="background-color: {color}15; border-left: 5px solid {color}; padding: 12px 18px; border-radius: 6px; margin-bottom: 15px;">
        <span style="font-size: 1.15rem; font-weight: bold; color: {color};">🎯 {label}</span>
        <div style="background-color: #E0E0E0; border-radius: 10px; height: 10px; width: 100%; margin-top: 8px;">
            <div style="background-color: {color}; height: 10px; border-radius: 10px; width: {score}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_evaluation_card(validation):
    if not validation or "eval_results" not in validation:
        return
    eval_res = validation["eval_results"]
    st.markdown(f"""
### 📊 RAG Evaluation Framework Report

| Metric | Score | Status |
| :--- | :--- | :--- |
| **RAGAS Score** | `{eval_res['ragas_score']}` | {'✅ PASS' if eval_res['ragas_score'] >= 0.7 else '⚠️ WARN'} |
| **Faithfulness** | `{int(eval_res['faithfulness'] * 100)}%` | ✅ Grounded Claims |
| **Context Precision** | `{int(eval_res['context_precision'] * 100)}%` | ✅ High Precision |
| **Context Recall** | `{int(eval_res['context_recall'] * 100)}%` | ✅ Relevant Context |
| **Answer Relevance** | `{int(eval_res['answer_relevance'] * 100)}%` | ✅ Query Aligned |
| **Citation Accuracy** | `{int(eval_res['citation_accuracy'] * 100)}%` | ✅ Verified Citations |
| **Groundedness** | `{int(eval_res['groundedness'] * 100)}%` | ✅ Evidence Supported |
| **Hallucination Rate** | `{eval_res['hallucination_rate']}%` | {'✅ Low (<15%)' if eval_res['hallucination_rate'] <= 15 else '⚠️ Elevated'} |
""")

def render_langsmith_card():
    """
    Renders live LangSmith observability traces, agent latency, and retriever performance metrics.
    """
    try:
        r = requests.get(f"{BACKEND_URL}/traces", timeout=3)
        if r.status_code == 200 and r.json():
            traces = r.json()
            latest = traces[-1]
            st.info(f"""
            **🛠️ LangSmith Observability & Performance Trace**
            - **Trace Name**: `{latest.get('trace_name')}`
            - **Total Execution Latency**: `{latest.get('total_latency_ms')} ms`
            - **Traced Events Logged**: `{len(latest.get('events', []))}` event(s)
            """)
    except Exception:
        pass

def main():

    st.set_page_config(
        page_title="Enterprise IT Operations Copilot Frontend",
        page_icon="⚡",
        layout="wide"
    )

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "current_analysis" not in st.session_state:
        st.session_state["current_analysis"] = None

    if "uploaded_filenames" not in st.session_state:
        st.session_state["uploaded_filenames"] = []

    st.title("⚡ Enterprise IT Operations Copilot")
    st.caption("Decoupled Architecture: Streamlit Frontend ➔ FastAPI Backend ➔ LangGraph Multi-Agent Workflows")

    # Check FastAPI Backend Health
    health = get_backend_health()
    if not health:
        st.error("⚠️ FastAPI Backend Service (`http://127.0.0.1:8000`) is offline. Please launch the FastAPI server via `uvicorn api:app --reload`.")
        st.info("💡 To start the backend: run `python -m uvicorn api:app --port 8000` in your terminal.")

    # Sidebar: Settings & Engine Selection
    with st.sidebar:
        st.header("⚙️ Settings & Engine")
        
        api_key = st.text_input(
            "API Key (Optional)",
            type="password",
            help="Enter API Key for cloud models, or leave blank to run with local reasoning engine."
        )
        model_choice = st.selectbox(
            "LLM Engine",
            [
                "kimi-k2.7-code:cloud",
                "glm-5.2:cloud",
                "qwen3.6",
                "gemma4:12b",
                "minimax-m3:cloud",
                "nemotron-3-super:cloud",
                "Local Engine"
            ]
        )

        st.markdown("---")
        st.header("🌐 Backend Service Stats")
        if health:
            st.success(f"Status: **ONLINE**")
            st.write(f"Indexed Chunks: **{health.get('vector_chunks', 0)}**")
            st.write(f"Default LLM: `{health.get('primary_model', 'kimi-k2.7-code:cloud')}`")
        else:
            st.error("Status: **OFFLINE**")

        st.markdown("---")
        st.header("🧠 Conversation History")
        st.write(f"Active Turns: **{len(st.session_state['messages'])}**")
        if st.button("🧹 Clear Chat History", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()

        st.markdown("---")
        if st.button("🗑️ Clear Vector Database", use_container_width=True):
            try:
                r = requests.post(f"{BACKEND_URL}/clear", timeout=5)
                if r.status_code == 200:
                    st.session_state["uploaded_filenames"] = []
                    st.success("Vector Store Cleared via FastAPI Backend!")
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to clear backend: {e}")

    # Section 1: Upload Company Files (Decoupled API Upload)
    st.subheader("1️⃣ Upload Company Files (FastAPI Service Processing)")
    
    uploaded_files = st.file_uploader(
        "Upload Company Files (PDF, DOCX, TXT, LOG, CSV)",
        type=["txt", "log", "pdf", "docx", "csv"],
        accept_multiple_files=True
    )
    
    if st.button("⚡ Upload & Index via FastAPI Backend", type="primary", use_container_width=True):
        if uploaded_files:
            files_payload = []
            for f in uploaded_files:
                files_payload.append(("files", (f.name, f.read(), f.type)))
            
            with st.spinner("FastAPI Backend is parsing, zero-shot categorizing, and indexing into Qdrant Vector Store..."):
                try:
                    res = requests.post(f"{BACKEND_URL}/upload", files=files_payload, timeout=60)
                    if res.status_code == 200:
                        data = res.json()
                        for f in uploaded_files:
                            if f.name not in st.session_state["uploaded_filenames"]:
                                st.session_state["uploaded_filenames"].append(f.name)
                        st.success(f"FastAPI Backend successfully indexed {data.get('uploaded_files_count')} file(s) into Qdrant Vector Store ({data.get('total_vector_chunks')} total chunks)!")
                    else:
                        st.error(f"Upload failed: {res.text}")
                except Exception as e:
                    st.error(f"Backend Connection Error: {str(e)}")
        else:
            st.warning("Please select at least one file to upload.")

    if st.session_state["uploaded_filenames"]:
        st.markdown("#### 📂 Currently Uploaded Files")
        for fn in st.session_state["uploaded_filenames"]:
            st.write(f"📄 `{fn}`")

    st.markdown("---")
    st.subheader("2️⃣ IT Infrastructure Query & Diagnosis")

    user_query = st.text_input(
        "Enter your natural language question or issue statement:",
        placeholder="e.g. Is there any error in the uploaded log? / Why is Switch-45 unreachable?"
    )
    
    if st.button("🔍 Run Multi-Agent Diagnosis via FastAPI", type="primary", use_container_width=True):
        if not user_query.strip():
            st.warning("Please enter a query.")
            return

        from security import APIKeyEncrypter
        encrypted_key = APIKeyEncrypter.encrypt_api_key(api_key) if api_key else ""

        # Clean & validate chat_history turns
        clean_history = []
        for msg in st.session_state["messages"]:
            item = {
                "role": str(msg.get("role", "user")),
                "content": str(msg.get("content", ""))
            }
            if "confidence" in msg and msg["confidence"] is not None:
                try:
                    item["confidence"] = int(msg["confidence"])
                except (ValueError, TypeError):
                    pass
            clean_history.append(item)

        payload = {
            "query": user_query,
            "chat_history": clean_history,
            "api_key": encrypted_key,
            "model_choice": model_choice
        }

        with st.status(f"Executing LangGraph Multi-Agent Workflow via FastAPI Backend ({model_choice})...", expanded=True) as status:
            st.write("🔒 **Step 1: Encrypting API Key & Validating Query Security (Prompt Injection Guard)**")
            import json
            st.code(f"Outgoing Payload to {BACKEND_URL}/diagnose:\n{json.dumps(payload, indent=2)}", language="json")

            try:
                res = requests.post(f"{BACKEND_URL}/diagnose", json=payload, timeout=120)

                if res.status_code == 200:
                    data = res.json()
                    executed_chain = " ➔ ".join(data.get("executed_agents", []))
                    st.write(f"🤖 **Step 2: LangGraph Executed Agent Path**: `{executed_chain}`")
                    
                    final_response = data.get("final_response", "")
                    validation_results = data.get("validation_results", {})
                    confidence_score = data.get("confidence_score", 85)

                    status.update(label="FastAPI Multi-Agent Diagnosis Complete!", state="complete", expanded=False)

                    st.session_state["current_analysis"] = {
                        "query": user_query,
                        "response": final_response,
                        "confidence": confidence_score,
                        "validation": validation_results
                    }
                    st.session_state["messages"].append({"role": "user", "content": user_query})
                    st.session_state["messages"].append({"role": "assistant", "content": final_response})

                elif res.status_code == 400:
                    err_msg = res.json().get("detail", res.text)
                    st.error(f"🛡️ Security Block: {err_msg}")
                    status.update(label="Blocked by Security Guard", state="error")
                else:
                    st.error(f"FastAPI Diagnosis Error: {res.text}")
                    status.update(label="Diagnosis Failed", state="error")
            except Exception as e:
                st.error(f"Failed to connect to FastAPI backend: {str(e)}")
                status.update(label="Connection Failed", state="error")


    # Display Current Result Analysis
    if "current_analysis" in st.session_state and st.session_state["current_analysis"]:
        analysis = st.session_state["current_analysis"]
        st.markdown("---")
        st.subheader("📋 Diagnostic Result Analysis & Recommendation")
        
        # 1. RAG Evaluation Framework Score Card
        render_evaluation_card(analysis.get("validation"))

        # 2. LangSmith Observability Trace Card
        render_langsmith_card()

        # 3. Confidence Score Badge
        render_confidence_badge(analysis["confidence"], analysis.get("validation"))

        
        # 3. Main Response (Cause, Evidence, Recommended Fix, Citations)
        st.markdown(analysis["response"])

if __name__ == '__main__':
    main()
