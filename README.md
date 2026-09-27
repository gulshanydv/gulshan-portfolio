# Gulshan Yadav — AI Engineer Portfolio & LangGraph Career Assistant

A modern, high-performance portfolio and production-ready **Agentic Career AI Assistant** built for **Gulshan Yadav** (AI Engineer specializing in Python Backend Development, Document Intelligence, PTAX AI Analysis, and RAG Architectures).

---

## 🌟 Key Features

1. **Expansive 3D Video Preloader**:
   - Full-page seamless video sequence with synchronized percentage counter (0% → 100%).
   - Dynamic slide-in of `GULSHAN YADAV` from the left at the 50% mark.
   - Smooth curtain-pull exit transitioning into the Hero section.

2. **Interactive Focus-Zoom Architecture**:
   - Dynamic hover spotlighting: hovering any capability card smoothly zooms in the active card while scaling down surrounding cards.

3. **LangGraph Career Assistant (Gulshan AI)**:
   - Built with **LangGraph `StateGraph`**, **LangChain**, and **Groq LLM API** (`qwen/qwen3.8-27b`).
   - Grounded strictly in Gulshan Yadav's verified resume and portfolio metrics.
   - High-performance guardrail routing node preventing off-topic queries, code generation, and prompt injections.

---

## 🧠 Chatbot Architecture (LangGraph + Groq)

```
                       [User Query]
                            │
                            ▼
               ┌─────────────────────────┐
               │  guardrail_router_node  │
               └────────────┬────────────┘
                            │
              ┌─────────────┴─────────────┐
        [Valid Career]               [Code/Off-topic/Injection]
              ▼                                   ▼
   ┌──────────────────────┐             ┌────────────────────┐
   │  career_agent_node   │             │    refusal_node    │
   │  (ChatGroq + Context)│             │ (Standard Refusal) │
   └──────────┬───────────┘             └─────────┬──────────┘
              │                                   │
              └─────────────┬─────────────────────┘
                            ▼
                         [ END ]
```

### Strict Guardrail Policies
- **Domain Restricted:** Only answers questions about Gulshan Yadav's career, AI systems, skills, metrics, education, and contact details.
- **Zero Code Generation:** Refuses requests to write code, build scripts, or debug arbitrary user code.
- **Prompt Injection Defense:** Rejects instructions to bypass rules, adopt alternative personas (e.g. DAN), or reveal system prompts.
- **Refusal Message:**
  > *"I am Gulshan's Career AI assistant. I can only answer questions about Gulshan Yadav's professional background, AI systems, experience, and projects. Please feel free to ask about his work!"*

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10+
- Groq API Key ([console.groq.com](https://console.groq.com))

### 2. Setup Virtual Environment & Dependencies
```bash
# Create virtual environment (if not already created)
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install fastapi uvicorn pydantic python-dotenv langchain-groq langgraph langchain-core groq pypdf
```

### 3. Configure Environment Variables
Create or edit `.env` in the root directory:
```env
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
GROQ_MODEL=qwen/qwen3.8-27b
PORT=8001
```

### 4. Run the Chatbot Backend
```bash
python chatbot_server.py
```
Backend will start on `http://127.0.0.1:8001`.

### 5. Run the Portfolio Frontend
```bash
python -m http.server 8000 --directory dist
```
Open `http://localhost:8000` in your web browser.

---

## 🧪 Automated End-to-End Testing

To run the automated E2E test suite covering valid career questions, multi-turn conversation memory, and guardrail enforcement:

```bash
python test_chatbot_e2e.py
```

---

## 📁 Repository Structure

```
├── assets/                  # 3D videos, avatars, and resume PDF (Gulshan-Yadav-Resume.pdf)
├── chatbot_server.py        # LangGraph StateGraph backend API for Gulshan AI Assistant
├── index.html               # Main portfolio application
├── dist/                    # Static distribution files
├── extracted_resume.txt     # Ground truth resume source text
├── test_chatbot_e2e.py      # End-to-end Python test suite for Chatbot API
└── README.md                # Project documentation
```
