# 🧠 Synapse Agent: 100% Private Deep Research Assistant

Synapse Agent is a sophisticated, fully private deep research assistant designed to run entirely on your local machine. It allows you to upload documents and engage in a deep, context-aware dialogue, with an AI that performs multi-step reasoning to provide comprehensive, synthesized answers.

This project was developed for the **#SRMHacksWithCodemate** Hackathon, specifically addressing the "Deep Researcher Agent" problem statement by creating a solution that handles all embedding, reasoning, and generation locally, without relying on external APIs. It builds upon the principles of local-first AI that I explored in my **Gemma Web** project, scaling up the capabilities to a more powerful server-side environment.

## ✨ Key Features

* **100% Local & Private**: All processing, from document analysis to AI reasoning, happens on your machine. Your data never leaves your computer.
* **Advanced Multi-Step Reasoning Pipeline**: The agent doesn't just find information; it *researches*. For each query, it performs:
   1. **Query Decomposition**: Breaks a complex question into smaller, manageable sub-questions.
   2. **Information Extraction**: Retrieves relevant data for each sub-question from your documents.
   3. **Synthesis**: Synthesizes the extracted information into a single, coherent answer.
   4. **Validation**: Cross-references the final answer against the source documents to ensure accuracy.
* **Transparent Research Process**: The user interface allows you to expand a "View Research Process" section to see exactly how the agent arrived at its conclusion, step-by-step.
* **Interactive Query Refinement**: After providing an answer, the agent suggests intelligent follow-up questions to help you dig deeper into your research.
* **Multi-Format Report Export**: Export your full research session, including the final answer, sources, and reasoning steps, into beautifully formatted **PDF**, **Markdown**, or **HTML** reports.
* **Robust Client-Server Architecture**: A responsive **Streamlit** frontend communicates with a powerful **FastAPI** backend, which handles all the heavy lifting in the background.

## 🛠️ Tech Stack

* **Backend**: FastAPI, Python
* **Frontend**: Streamlit
* **LLM Integration**: `llama-cpp-python` running a GGUF version of `Mistral-7B-Instruct-v0.2`
* **Vector Embeddings & Search**: `sentence-transformers` & `FAISS`
* **Document Processing**: `langchain`, `pypdf`, `python-docx`
* **Report Generation**: `fpdf2`, `markdown`

## 🚀 Getting Started

Follow these instructions to set up and run Synapse Agent on your local machine.

### Prerequisites

* Python 3.9+
* Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/sanalokeshreddy/synapse-agent.git
cd synapse-agent
```

### Step 2: Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
# Create the environment
python -m venv venv_stable

# Activate the environment
# On Windows (PowerShell):
.\venv_stable\Scripts\Activate.ps1

# On macOS/Linux:
source venv_stable/bin/activate
```

### Step 3: Install Dependencies

Install all the required Python packages using pip.

```bash
pip install -r requirements.txt
```

### Step 4: Download the AI Model

Before you can run the agent, you need to download the local LLM. This script will download the `mistral-7b-instruct-v0.2.Q2_K.gguf` model (~2.4 GB).

```bash
python download_model.py
```

### Step 5: Run the Application (Two Terminals Required)

Synapse Agent runs as two separate processes: the backend server and the frontend interface. You will need to open **two separate terminals** for this.

#### Terminal 1: Start the FastAPI Backend

In your first terminal (with the virtual environment activated), run the following command to start the backend server:

```bash
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8080
```

#### Terminal 2: Start the Streamlit Frontend

In your second terminal (also with the virtual environment activated), run this command to start the user interface:

```bash
streamlit run app.py
```

Once both servers are running, open your web browser and navigate to the URL provided by Streamlit (typically `http://localhost:8501`) to start using Synapse Agent!
