import time
import streamlit as st

from utils import parse_uploaded_file, detect_document_category, FAISSVectorStore
from agents import (
    DocumentationAgent,
    NetworkAgent,
    LogAnalysisAgent,
    IncidentAgent,
    PlannerAgent,
    SupervisorAgent,
    build_copilot_graph
)


# ==============================================================================
# PROJECT 3: ENTERPRISE IT OPERATIONS COPILOT (OPENAI / LOCAL ENGINE)
# ==============================================================================


# Initialize FAISS Vector Store
@st.cache_resource
def get_faiss_vector_store():
    return FAISSVectorStore()

def sync_vector_index():
    vs = get_faiss_vector_store()

    if "knowledge_base" not in st.session_state:
        st.session_state["knowledge_base"] = []

    try:
        vs.build_index(st.session_state["knowledge_base"])
    except Exception as e:
        st.error(f"Vector index build failed: {e}")

    return vs

def extract_page_text(p):
    if isinstance(p, dict):
        return p.get('content', '')
    elif isinstance(p, str):
        return p
    return str(p)

def render_confidence_badge(score, validation=None):
    """
    Renders visual confidence score indicator with engineering thresholds & validation details.
    """
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

def render_validation_card(validation):
    """
    Renders Diagnostic Validation Layer Audit results.
    """
    if not validation:
        return
    fault_str = "⚠️ Error/Fault Keywords Detected" if validation.get("is_actual_error") else "✅ Operational Status Normal (No Error Keywords)"
    suff_str = "✅ Sufficient Context" if validation.get("is_evidence_sufficient") else "⚠️ Insufficient Context"
    category = validation.get("error_category", "General Technical Analysis")

    st.info(f"""
    **🛡️ Diagnostic Validation Layer Audit**
    - **Fault Detected**: {fault_str}
    - **Evidence Sufficiency**: {suff_str}
    - **Category**: `{category}`
    """)

# Main App
def main():
    st.set_page_config(
        page_title="Enterprise IT Operations Copilot",
        page_icon="⚡",
        layout="wide"
    )

    # Initialize Session State inside main() execution flow
    if "knowledge_base" not in st.session_state:
        st.session_state["knowledge_base"] = []

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "current_analysis" not in st.session_state:
        st.session_state["current_analysis"] = None

    st.title("⚡ Enterprise IT Operations Copilot")
    st.caption("FAISS Embeddings RAG + Multi-Agent AI System with Auto-Category Detection & Validation Layer")

    # Sync FAISS vector store with active knowledge base
    vector_store = sync_vector_index()

    # Sidebar: Settings & Model Selection
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
        st.header("🧠 Conversation History")
        st.write(f"Active Turns: **{len(st.session_state['messages'])}**")
        if st.button("🧹 Clear Chat History", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()

        st.markdown("---")
        st.header("📁 Knowledge Base Stats")
        st.write(f"Uploaded Files: **{len(st.session_state['knowledge_base'])}**")
        st.write(f"FAISS Chunks: **{len(vector_store.chunks)}**")
        if st.button("🗑️ Clear Uploaded Files", use_container_width=True):
            st.session_state["knowledge_base"] = []
            sync_vector_index()
            st.rerun()

    # Section 1: Upload Company Files (Auto-Detect Category)
    st.subheader("1️⃣ Upload Company Files (Auto-Category Detection)")
    
    uploaded_files = st.file_uploader(
        "Upload Company Files (PDF, DOCX, TXT, LOG, CSV)",
        type=["txt", "log", "pdf", "docx", "csv"],
        accept_multiple_files=True
    )
    
    if st.button("⚡ Auto-Detect Category, Chunk & Index into FAISS", type="primary", use_container_width=True):
        if uploaded_files:
            count = 0
            for f in uploaded_files:
                pages = parse_uploaded_file(f)
                full_text = "\n".join([extract_page_text(p) for p in pages])
                total_chars = len(full_text)
                
                # Auto-detect category
                auto_category = detect_document_category(f.name, full_text)
                
                new_id = len(st.session_state["knowledge_base"]) + 1
                st.session_state["knowledge_base"].append({
                    "id": new_id,
                    "source_type": auto_category,
                    "title": f.name,
                    "pages": pages,
                    "content": full_text,
                    "total_pages": len(pages),
                    "total_chars": total_chars
                })
                count += 1
            sync_vector_index()
            st.success(f"Successfully auto-categorized & indexed {count} file(s) into FAISS Vector Store ({len(vector_store.chunks)} total chunks)!")
            st.rerun()
        else:
            st.warning("Please select at least one file to upload.")

    # Display Currently Uploaded Files Only with Auto-Detected Category
    if st.session_state["knowledge_base"]:
        st.markdown("#### 📂 Currently Uploaded & Auto-Categorized Files")
        for idx, item in enumerate(st.session_state["knowledge_base"]):
            c1, c2 = st.columns([4, 1])
            with c1:
                p_cnt = item.get('total_pages', 1)
                st.write(f"🏷️ **Auto-Category: [{item['source_type']}]** | `{item['title']}` — {p_cnt} page(s) | {item.get('total_chars', len(item.get('content', '')))} characters")
            with c2:
                if st.button("❌ Remove", key=f"del_{idx}"):
                    st.session_state["knowledge_base"].pop(idx)
                    sync_vector_index()
                    st.rerun()
    else:
        st.info("ℹ️ No company files uploaded yet. Upload files above to automatically detect their categories and index into FAISS.")

    st.markdown("---")
    st.subheader("2️⃣ IT Infrastructure Query & Diagnosis")

    user_query = st.text_input(
        "Enter your natural language question or issue statement:",
        placeholder="e.g. Is there any error in the uploaded log? / Why is Switch-45 unreachable?"
    )
    
    if st.button("🔍 Run Multi-Agent Diagnosis", type="primary", use_container_width=True):
        if not st.session_state["knowledge_base"]:
            st.error("No company files uploaded. Please upload your company files above first.")
            return

        if not user_query.strip():
            st.warning("Please enter a query.")
            return

        with st.status(f"Executing LangGraph Multi-Agent Workflow via {model_choice}...", expanded=True) as status:
            st.write("🌐 **Step 1: Instantiating LangGraph Multi-Agent Supervisor Graph**")
            copilot_graph = build_copilot_graph(vector_store)
            
            initial_state = {
                "query": user_query,
                "chat_history": st.session_state["messages"],
                "doc_evidence": [],
                "net_evidence": [],
                "log_evidence": [],
                "inc_evidence": [],
                "validation_results": {},
                "confidence_score": 0,
                "executed_agents": [],
                "next_agent": "Supervisor",
                "final_response": "",
                "api_key": api_key,
                "model_choice": model_choice
            }
            
            st.write("🤖 **Step 2: Supervisor Routing & Sub-Agent Execution**")
            final_state = copilot_graph.invoke(initial_state)
            
            executed_chain = " ➔ ".join(final_state.get("executed_agents", []))
            st.write(f"  • Execution Path: `{executed_chain}`")
            
            final_response = final_state.get("final_response", "Diagnosis synthesis failed.")
            validation_results = final_state.get("validation_results", {})
            confidence_score = final_state.get("confidence_score", 85)

            st.write(f"🧠 **Step 3: Diagnostic Validation & Synthesis Complete**")
            status.update(label="LangGraph Workflow Execution Complete!", state="complete", expanded=False)


        # Store result in session state to display on current page
        st.session_state["current_analysis"] = {
            "query": user_query,
            "response": final_response,
            "confidence": confidence_score,
            "validation": validation_results
        }
        st.session_state["messages"].append({"role": "user", "content": user_query})
        st.session_state["messages"].append({"role": "assistant", "content": final_response, "confidence": confidence_score})

    # Display Current Result Analysis directly on Page
    if "current_analysis" in st.session_state and st.session_state["current_analysis"]:
        analysis = st.session_state["current_analysis"]
        st.markdown("---")
        st.subheader("📋 Diagnostic Result Analysis & Recommendation")
        
        # 1. Validation Layer Audit Card
        render_validation_card(analysis.get("validation"))

        # 2. Confidence Score Badge
        render_confidence_badge(analysis["confidence"], analysis.get("validation"))
        
        # 3. Main Response (Cause, Evidence, Recommended Fix, Citations)
        st.markdown(analysis["response"])

if __name__ == '__main__':
    main()
