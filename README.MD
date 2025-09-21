🧠 Synapse Agent: Deep Research Assistant
Synapse Agent is a powerful, locally-run AI research assistant. It leverages a Retrieval-Augmented Generation (RAG) pipeline to analyze documents you upload, answer complex questions, and generate comprehensive research reports—all without your data ever leaving your machine.

This project is a submission for the CodeMate Hackathon, built to satisfy the "Deep Researcher Agent" problem statement, emphasizing local embedding, multi-step reasoning, and zero reliance on external APIs.


Key Features
100% Local & Private: All models (for embeddings and generation) run directly on your machine. Your documents and queries are never sent to the cloud.

Deep Research Engine: Implements a multi-step research process: query decomposition, source identification, information extraction, and synthesis.

Document Agnostic: Supports PDF, TXT, and DOCX files for analysis.

Interactive UI: A user-friendly interface built with Streamlit for uploading documents and having conversations with the agent.

Exportable Reports: Generate and export detailed research reports in PDF, Markdown, or HTML formats.

Smart Query Refinement: Suggests intelligent follow-up questions to help you dig deeper into your research.

Tech Stack
Backend: FastAPI, Llama-cpp-python

Frontend: Streamlit

AI Models:

all-MiniLM-L6-v2 (for sentence embeddings)

Mistral-7B-Instruct-v0.2 (GGUF for LLM reasoning)

Vector Store: FAISS (for efficient similarity search)

🚀 Getting Started
Follow these steps to set up and run Synapse Agent on your local machine.

Step 1: Clone the Repository
git clone [https://github.com/your-username/synapse_agent.git](https://github.com/your-username/synapse_agent.git)
cd synapse_agent

Step 2: Download the AI Model
This project requires a local Large Language Model. Run the provided script to download the necessary model file (~2.4 GB).

python download_model.py

Step 3: Create and Activate a Virtual Environment
It's highly recommended to use a virtual environment to manage project dependencies.

# Create the virtual environment
python -m venv venv_stable

# Activate it
# On Windows (PowerShell):
.\venv_stable\Scripts\Activate.ps1
# On macOS/Linux:
source venv_stable/bin/activate

Step 4: Install Dependencies
Install all the required Python packages.

pip install -r requirements.txt

Step 5: Run the Application
The application consists of two parts: the FastAPI backend and the Streamlit frontend. You need to run them in two separate terminals.

Terminal 1: Start the FastAPI Backend

# Make sure your virtual environment is active
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8080

Terminal 2: Start the Streamlit Frontend

# Make sure your virtual environment is active in this terminal too
streamlit run app.py

Once both are running, your web browser should open to the Streamlit interface, and you can start using the Synapse Agent!