# main.py (Fixed)
import os
import faiss
import re
import asyncio
import json
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np
from llama_cpp import Llama
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fpdf import FPDF

# Import our new modules
from fastapi.middleware.cors import CORSMiddleware
from research_agent import DeepResearchAgent
from document_processor import DocumentProcessor
from report_generator import ReportGenerator
from query_refinement import QueryRefiner

# --- Global state for tracking processing ---
processing_status = {}

# --- WebSocket Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_json(self, data: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except:
                # Connection might be closed
                pass

manager = ConnectionManager()
app = FastAPI()
retriever_model, llm, loaded_indices = None, None, {}
research_agent, document_processor, report_generator, query_refiner = None, None, None, None

# --- On-Demand Model Loading ---
def get_retriever():
    global retriever_model
    if retriever_model is None:
        retriever_model = SentenceTransformer('all-MiniLM-L6-v2')
    return retriever_model

def get_llm():
    global llm
    if llm is None:
        # Using highly compressed Mistral-7B Q2_K for much better resource usage
        llm = Llama(
            model_path="./mistral-7b-instruct-v0.2.Q2_K.gguf", 
            n_ctx=10000,
            verbose=False,
            n_threads=4,
            n_batch=128,  # Even smaller batch size
            n_gpu_layers=0
        )
    return llm

def get_research_agent():
    global research_agent
    if research_agent is None:
        research_agent = DeepResearchAgent(get_llm(), get_retriever())
    return research_agent

def get_document_processor():
    global document_processor
    if document_processor is None:
        document_processor = DocumentProcessor()
    return document_processor

def get_report_generator():
    global report_generator
    if report_generator is None:
        report_generator = ReportGenerator()
    return report_generator

def get_query_refiner():
    global query_refiner
    if query_refiner is None:
        query_refiner = QueryRefiner(get_llm())
    return query_refiner

# --- Background Document Processing ---
async def process_document_background(filepath: str, filename: str):
    base_filename = os.path.splitext(filename)[0]
    processing_status[base_filename] = {"status": "processing", "message": "Starting document processing..."}
    
    try:
        await manager.send_json({"status": "Step 1/4: Reading and cleaning document..."})
        processing_status[base_filename] = {"status": "processing", "message": "Step 1/4: Reading and cleaning document..."}
        
        processor = get_document_processor()
        try:
            document_data = processor.process_document(filepath)
        except Exception as e:
            error_msg = f"Error processing document: {str(e)}"
            await manager.send_json({"status": error_msg})
            processing_status[base_filename] = {"status": "error", "message": error_msg}
            return
        
        await manager.send_json({"status": "Step 2/4: Creating embeddings..."})
        processing_status[base_filename] = {"status": "processing", "message": "Step 2/4: Creating embeddings..."}
        
        retriever = get_retriever()
        embeddings = retriever.encode(document_data["chunks"], show_progress_bar=False)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        
        await manager.send_json({"status": "Step 3/4: Saving processed data..."})
        processing_status[base_filename] = {"status": "processing", "message": "Step 3/4: Saving processed data..."}
        
        processed_dir = "processed_docs"
        os.makedirs(processed_dir, exist_ok=True)
        
        faiss.write_index(index, os.path.join(processed_dir, f"{base_filename}.index"))
        with open(os.path.join(processed_dir, f"{base_filename}.chunks.txt"), 'w', encoding='utf-8') as f:
            for chunk in document_data["chunks"]:
                f.write(f"{chunk.strip()}\n")
        
        # Save document metadata
        with open(os.path.join(processed_dir, f"{base_filename}.meta.json"), 'w', encoding='utf-8') as f:
            json.dump(document_data["metadata"], f)
        
        await manager.send_json({"status": "Step 4/4: Finalizing..."})
        processing_status[base_filename] = {"status": "processing", "message": "Step 4/4: Finalizing..."}
        
        # Mark as complete
        processing_status[base_filename] = {"status": "complete", "message": "Document processing completed successfully"}
        await manager.send_json({"status": "complete", "document": filename})
        
        print(f"Background task finished for: {filename}")
        
    except Exception as e:
        error_msg = f"Processing failed: {str(e)}"
        processing_status[base_filename] = {"status": "error", "message": error_msg}
        await manager.send_json({"status": error_msg})
        print(f"Error processing {filename}: {e}")

# --- API Endpoints ---
@app.post("/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    filepath = os.path.join(temp_dir, file.filename)
    with open(filepath, "wb") as buffer:
        buffer.write(await file.read())
    
    # Initialize processing status
    base_filename = os.path.splitext(file.filename)[0]
    processing_status[base_filename] = {"status": "queued", "message": "Processing queued"}
    
    background_tasks.add_task(process_document_background, filepath, file.filename)
    return {"message": "Processing started."}

@app.get("/check_processing_status/{filename}")
async def check_processing_status(filename: str):
    """Check the processing status of a document"""
    if filename in processing_status:
        return processing_status[filename]
    else:
        # Check if processed files exist
        processed_dir = "processed_docs"
        index_path = os.path.join(processed_dir, f"{filename}.index")
        chunks_path = os.path.join(processed_dir, f"{filename}.chunks.txt")
        
        if os.path.exists(index_path) and os.path.exists(chunks_path):
            return {"status": "complete", "message": "Document processing completed"}
        else:
            return {"status": "not_found", "message": "Document not found or not processed"}

@app.websocket("/ws/status")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

class Query(BaseModel):
    text: str
    document_name: str
    research_depth: str = "standard"  # Options: quick, standard, deep

class ResearchRequest(BaseModel):
    query: Query
    follow_up_context: Dict[str, Any] = None

class ChatHistory(BaseModel):
    messages: List[Dict[str, Any]]
    document_name: str
    format: str = "pdf"  # Options: pdf, markdown, html

async def stream_agent_response(query: Query):
    base_filename = os.path.splitext(query.document_name)[0]
    index_path = f"processed_docs/{base_filename}.index"
    chunks_path = f"processed_docs/{base_filename}.chunks.txt"

    yield json.dumps({"type": "status", "data": f"Loading index for '{query.document_name}'..."}) + "\n"
    
    # Check if files exist
    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        yield json.dumps({"type": "error", "data": f"Document '{query.document_name}' not found or not processed"}) + "\n"
        return
    
    if index_path not in loaded_indices:
        loaded_indices[index_path] = faiss.read_index(index_path)
    index = loaded_indices[index_path]
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = [line.strip() for line in f.readlines()]

    yield json.dumps({"type": "status", "data": "Loading models..."}) + "\n"
    retriever = get_retriever()
    
    yield json.dumps({"type": "status", "data": "Searching for relevant sources..."}) + "\n"
    query_embedding = retriever.encode([query.text])
    _, indices = index.search(np.array(query_embedding).astype('float32'), k=10)  # Get more chunks for research
    retrieved_chunks = [chunks[i] for i in indices[0]]
    
    # Use research agent for deep research
    if query.research_depth != "quick":
        yield json.dumps({"type": "status", "data": "Conducting deep research..."}) + "\n"
        research_agent = get_research_agent()
        research_result = research_agent.conduct_research(query.text, retrieved_chunks)
        
        yield json.dumps({"type": "sources", "data": retrieved_chunks}) + "\n"
        yield json.dumps({"type": "research_steps", "data": research_result["research_steps"]}) + "\n"
        
        # Stream the answer
        for sentence in research_result["answer"].split('. '):
            if sentence.strip():
                yield json.dumps({"type": "token", "data": sentence + '. '}) + "\n"
        
        yield json.dumps({"type": "validation", "data": research_result["validation"]}) + "\n"
        
        # Suggest follow-up questions
        query_refiner = get_query_refiner()
        follow_ups = query_refiner.suggest_follow_up_questions(query.text, research_result["answer"])
        yield json.dumps({"type": "follow_up_questions", "data": follow_ups}) + "\n"
    else:
        # Standard RAG approach for quick queries
        context = "\n".join(retrieved_chunks)
        prompt = f"[INST] Use the context to answer the question.\nContext: {context}\nQuestion: {query.text} [/INST]"
        
        yield json.dumps({"type": "status", "data": "Generating response..."}) + "\n"
        llm_model = get_llm()
        for chunk in llm_model(prompt, max_tokens=512, temperature=0.2, stream=True):
            token = chunk["choices"][0]["text"]
            yield json.dumps({"type": "token", "data": token}) + "\n"
        
        yield json.dumps({"type": "sources", "data": retrieved_chunks}) + "\n"

@app.post("/agent")
def run_agent_streaming(query: Query):
    return StreamingResponse(stream_agent_response(query), media_type="application/x-ndjson")

@app.post("/research")
def conduct_research(research_request: ResearchRequest):
    """Endpoint for conducting deep research with additional context"""
    base_filename = os.path.splitext(research_request.query.document_name)[0]
    index_path = f"processed_docs/{base_filename}.index"
    chunks_path = f"processed_docs/{base_filename}.chunks.txt"
    
    # Check if files exist
    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        raise HTTPException(status_code=404, detail="Document not found or not processed")
    
    if index_path not in loaded_indices:
        loaded_indices[index_path] = faiss.read_index(index_path)
    index = loaded_indices[index_path]
    
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = [line.strip() for line in f.readlines()]
    
    retriever = get_retriever()
    query_embedding = retriever.encode([research_request.query.text])
    _, indices = index.search(np.array(query_embedding).astype('float32'), k=15)
    retrieved_chunks = [chunks[i] for i in indices[0]]
    
    research_agent = get_research_agent()
    research_result = research_agent.conduct_research(research_request.query.text, retrieved_chunks)
    
    # Add follow-up questions
    query_refiner = get_query_refiner()
    follow_ups = query_refiner.suggest_follow_up_questions(
        research_request.query.text, 
        research_result["answer"]
    )
    research_result["follow_up_questions"] = follow_ups
    
    return research_result

@app.post("/refine_query")
def refine_query(query: Query, feedback: str):
    """Refine a query based on user feedback"""
    query_refiner = get_query_refiner()
    refined_query = query_refiner.refine_query(query.text, feedback)
    
    return {"original_query": query.text, "refined_query": refined_query}

@app.post("/export")
def export_chat_to_pdf(history: ChatHistory):
    pdf = FPDF()
    pdf.add_page()
    
    # Use a font that supports a wider range of characters
    pdf.set_font("Arial", size=16)
    
    # Add Title
    title = f"Synapse Agent: Conversation on '{history.document_name}'"
    pdf.cell(0, 10, txt=title.encode('latin-1', 'replace').decode('latin-1'), ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", size=12)
    # Define a consistent width for multi-cells (A4 width 210mm - 10mm margins)
    cell_width = 190 

    for message in history.messages:
        role = message.get("role")
        content = message.get("content", "")
        # Sanitize content to prevent encoding errors
        sanitized_content = content.encode('latin-1', 'replace').decode('latin-1')
        
        if role == "user":
            pdf.set_font("Arial", 'B', 12)
            pdf.multi_cell(cell_width, 8, f"You: {sanitized_content}")
            pdf.ln(5)
        elif role == "assistant":
            pdf.set_font("Arial", '', 12)
            pdf.multi_cell(cell_width, 8, f"Agent: {sanitized_content}")
            
            if "sources" in message and message["sources"]:
                pdf.set_font("Arial", 'I', 10)
                pdf.multi_cell(cell_width, 6, "--- Sources Used ---")
                for i, source in enumerate(message["sources"]):
                    sanitized_source = source.encode('latin-1', 'replace').decode('latin-1')
                    pdf.multi_cell(cell_width, 5, f"Source {i+1}: {sanitized_source[:250]}...")
                pdf.ln(5)
    
    export_dir = "temp_exports"
    os.makedirs(export_dir, exist_ok=True)
    pdf_path = os.path.join(export_dir, "chat_report.pdf")
    pdf.output(pdf_path)
    
    return FileResponse(path=pdf_path, media_type='application/pdf', filename='Synapse_Chat_Report.pdf')

@app.post("/export_research")
def export_research_report(research_data: Dict[str, Any], format: str = "pdf"):
    """Export a research report in the specified format"""
    report_generator = get_report_generator()
    export_dir = "temp_exports"
    os.makedirs(export_dir, exist_ok=True)
    
    if format == "pdf":
        output_path = os.path.join(export_dir, "research_report.pdf")
        report_generator.generate_pdf_report(research_data, output_path)
        return FileResponse(path=output_path, media_type='application/pdf', filename='Research_Report.pdf')
    elif format == "markdown":
        output_path = os.path.join(export_dir, "research_report.md")
        report_generator.generate_markdown_report(research_data, output_path)
        return FileResponse(path=output_path, media_type='text/markdown', filename='Research_Report.md')
    elif format == "html":
        output_path = os.path.join(export_dir, "research_report.html")
        report_generator.generate_html_report(research_data, output_path)
        return FileResponse(path=output_path, media_type='text/html', filename='Research_Report.html')
    else:
        raise HTTPException(status_code=400, detail="Unsupported format")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)