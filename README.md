# ⚡ Enterprise IT Operations Copilot

An AI-powered IT infrastructure diagnosis and telemetry copilot built with Streamlit, FAISS Vector Search RAG, and a Multi-Agent Architecture.

---

## 🌟 Key Features

- **⚡ Auto-Category Document Ingestion**: Upload PDF, DOCX, TXT, LOG, or CSV files. Automatically classifies documents into *SOP / Manual*, *Network Configuration*, *Server Log / Alert*, or *Incident Ticket*.
- **🔎 FAISS Vector Store RAG**: Chunks and indexes document text for fast semantic retrieval with source page and chunk citations.
- **🤖 Multi-Agent Architecture**:
  - **Documentation Agent**: Queries SOPs and manuals for standard recovery procedures.
  - **Network Agent**: Inspects switch and router topologies and VLAN configs.
  - **Log Analysis Agent**: Scans system logs, telemetry, and alert traces.
  - **Incident Agent**: Searches historical incident tickets and past resolution logs.
  - **Planner Agent**: Synthesizes evidence across sub-agents and produces actionable root-cause diagnostics.
- **🛡️ Diagnostic Validation Layer**: Audits retrieved chunks for explicit fault keywords, evidence sufficiency, error category determination, and confidence scoring.
- **⚙️ Dual LLM Engine**: Supports live OpenAI models (`gpt-4o-mini`, `gpt-4o`) with automatic fallback to a zero-dependency Local Reasoning Engine.

---

## 📁 Repository Structure

```
├── main.py               # Main entry point for Streamlit Cloud deployment
├── app.py                # Core Streamlit app logic & UI components
├── streamlit_app.py      # Streamlit Cloud launcher script
├── utils.py              # File parsing, document category detection, & FAISS vector store
├── validator.py          # Diagnostic Validation Layer audit engine
├── agents/               # Multi-agent implementations
│   ├── doc_agent.py
│   ├── network_agent.py
│   ├── log_agent.py
│   ├── incident_agent.py
│   └── planner_agent.py
├── .streamlit/
│   └── config.toml       # Streamlit server & theme configuration
└── requirements.txt      # Python dependencies
```

---

## 🚀 Getting Started

### 1. Local Setup

```bash
# Clone the repository
git clone https://github.com/Pujithsai7/Enterprise-IT-Operations-Copilot.git
cd Enterprise-IT-Operations-Copilot

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run main.py
```

### 2. Streamlit Cloud Deployment

1. Push your changes to GitHub.
2. Navigate to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create a new app and select your repository: `Pujithsai7/Enterprise-IT-Operations-Copilot`.
4. Set **Main file path** to `main.py`.
5. Click **Deploy!**

---

## 📄 License

MIT License
