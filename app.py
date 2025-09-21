# app.py (Fixed)
import streamlit as st
import requests
import json
import httpx
import time

# --- Page Configuration ---
st.set_page_config(page_title="Synapse Agent", page_icon="🧠", layout="wide")

# --- Main App ---
def main():
    st.title("🧠 Synapse Agent: Deep Research Assistant")
    st.markdown("Upload documents and ask questions to conduct deep research using local AI models.")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_document" not in st.session_state:
        st.session_state.current_document = None
    if "research_mode" not in st.session_state:
        st.session_state.research_mode = "standard"
    if "research_data" not in st.session_state:
        st.session_state.research_data = None
    if "processing" not in st.session_state:
        st.session_state.processing = False
    
    # Sidebar for document upload and settings
    with st.sidebar:
        st.header("📄 Document Management")
        uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt", "docx"])
        
        if uploaded_file and not st.session_state.processing:
            if st.button("Process Document"):
                process_document_sync(uploaded_file)
        
        if st.session_state.current_document:
            st.success(f"Current Document: {st.session_state.current_document}")
        
        if st.session_state.processing:
            st.info("Processing document... Please wait.")
        
        st.header("⚙️ Research Settings")
        research_depth = st.selectbox(
            "Research Depth",
            ["quick", "standard", "deep"],
            index=1,
            help="Quick: Fast response, Standard: Balanced, Deep: Comprehensive multi-step research"
        )
        
        st.session_state.research_mode = research_depth
        
        if st.session_state.research_data:
            st.header("📊 Export Options")
            export_format = st.selectbox("Export Format", ["pdf", "markdown", "html"])
            if st.button("Export Research Report"):
                with st.spinner("Generating report..."):
                    try:
                        response =                         requests.post(
                            "http://localhost:8080/export_research",
                            json=st.session_state.research_data,
                            params={"format": export_format}
                        )
                        if response.status_code == 200:
                            st.success("Report generated successfully!")
                            st.download_button(
                                label="Download Report",
                                data=response.content,
                                file_name=f"research_report.{export_format}",
                                mime=response.headers['content-type']
                            )
                        else:
                            st.error("Failed to generate report")
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        if st.session_state.messages:
            if st.button("Export Conversation"):
                with st.spinner("Generating PDF..."):
                    try:
                        response = requests.post(
                            "http://localhost:8080/export",
                            json={
                                "messages": st.session_state.messages,
                                "document_name": st.session_state.current_document or "Unknown Document",
                                "format": "pdf"
                            }
                        )
                        if response.status_code == 200:
                            st.success("PDF generated successfully!")
                            st.download_button(
                                label="Download PDF",
                                data=response.content,
                                file_name="Synapse_Chat_Report.pdf",
                                mime="application/pdf"
                            )
                        else:
                            st.error("Failed to generate PDF")
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    # Main chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("View Sources"):
                    for i, source in enumerate(message["sources"]):
                        st.markdown(f"**Source {i+1}:** {source[:200]}...")
            
            if message["role"] == "assistant" and "research_steps" in message:
                with st.expander("View Research Process"):
                    for step in message["research_steps"]:
                        st.markdown(f"**{step['description']}**")
                        st.markdown(step['result'])
                        st.divider()
            
            if message["role"] == "assistant" and "validation" in message:
                with st.expander("View Validation Report"):
                    st.markdown(message["validation"])
            
            if message["role"] == "assistant" and "follow_up_questions" in message:
                st.markdown("**Suggested Follow-up Questions:**")
                for question in message["follow_up_questions"]:
                    if st.button(question, key=f"followup_{hash(question)}"):
                        st.session_state.messages.append({"role": "user", "content": question})
                        st.rerun()
    
    # Query input
    if prompt := st.chat_input("Ask a question about your document"):
        if not st.session_state.current_document:
            st.warning("Please upload and process a document first.")
            st.stop()
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            sources = []
            research_steps = []
            validation = ""
            follow_up_questions = []
            
            try:
                # For deep research, use the research endpoint
                if st.session_state.research_mode == "deep":
                    research_request = {
                        "query": {
                            "text": prompt,
                            "document_name": st.session_state.current_document,
                            "research_depth": "deep"
                        }
                    }
                    
                    response = requests.post(
                        "http://localhost:8080/research",
                        json=research_request
                    )
                    
                    if response.status_code == 200:
                        research_data = response.json()
                        st.session_state.research_data = research_data
                        
                        full_response = research_data["answer"]
                        sources = research_data.get("research_results", [{}])[0].get("relevant_chunks", [])
                        research_steps = research_data.get("research_steps", [])
                        validation = research_data.get("validation", "")
                        follow_up_questions = research_data.get("follow_up_questions", [])
                        
                        message_placeholder.markdown(full_response)
                    else:
                        st.error("Research failed")
                
                # For standard and quick queries, use streaming
                else:
                    query_data = {
                        "text": prompt,
                        "document_name": st.session_state.current_document,
                        "research_depth": st.session_state.research_mode
                    }
                    
                    with httpx.Client() as client:
                        with client.stream(
                            "POST", 
                            "http://localhost:8080/agent", 
                            json=query_data,
                            timeout=300
                        ) as response:
                            if response.status_code == 200:
                                for line in response.iter_lines():
                                    if line:
                                        try:
                                            data = json.loads(line)
                                            if data["type"] == "token":
                                                full_response += data["data"]
                                                message_placeholder.markdown(full_response + "▌")
                                            elif data["type"] == "sources":
                                                sources = data["data"]
                                            elif data["type"] == "research_steps":
                                                research_steps = data["data"]
                                            elif data["type"] == "validation":
                                                validation = data["data"]
                                            elif data["type"] == "follow_up_questions":
                                                follow_up_questions = data["data"]
                                        except json.JSONDecodeError:
                                            continue
                                
                                message_placeholder.markdown(full_response)
                            else:
                                st.error("Query failed")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
            
            # Store the assistant's response
            assistant_message = {
                "role": "assistant", 
                "content": full_response,
                "sources": sources
            }
            
            if research_steps:
                assistant_message["research_steps"] = research_steps
            if validation:
                assistant_message["validation"] = validation
            if follow_up_questions:
                assistant_message["follow_up_questions"] = follow_up_questions
            
            st.session_state.messages.append(assistant_message)
            
            # Display sources if available
            if sources:
                with st.expander("View Sources"):
                    for i, source in enumerate(sources):
                        st.markdown(f"**Source {i+1}:** {source[:200]}...")
            
            # Display research steps if available
            if research_steps:
                with st.expander("View Research Process"):
                    for step in research_steps:
                        st.markdown(f"**{step['description']}**")
                        st.markdown(step['result'])
                        st.divider()
            
            # Display validation if available
            if validation:
                with st.expander("View Validation Report"):
                    st.markdown(validation)
            
            # Display follow-up questions if available
            if follow_up_questions:
                st.markdown("**Suggested Follow-up Questions:**")
                for question in follow_up_questions:
                    if st.button(question, key=f"followup_{hash(question)}"):
                        st.session_state.messages.append({"role": "user", "content": question})
                        st.rerun()

def process_document_sync(uploaded_file):
    """Synchronous document processing with status polling"""
    st.session_state.processing = True
    st.session_state.current_document = None
    
    # Upload file
    files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    
    with st.spinner("Uploading and processing document..."):
        try:
            # Upload the file
            response = requests.post("http://localhost:8080/upload", files=files, timeout=300)
            
            if response.status_code != 200:
                st.error("Upload failed")
                st.session_state.processing = False
                return
            
            # Poll for completion
            max_wait_time = 300  # 5 minutes
            poll_interval = 2  # 2 seconds
            elapsed_time = 0
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            while elapsed_time < max_wait_time:
                try:
                    # Check if processing is complete by trying to access the processed files
                    base_filename = uploaded_file.name.rsplit('.', 1)[0]
                    check_response = requests.get(
                        f"http://localhost:8080/check_processing_status/{base_filename}",
                        timeout=10
                    )
                    
                    if check_response.status_code == 200:
                        result = check_response.json()
                        if result.get("status") == "complete":
                            st.session_state.current_document = uploaded_file.name
                            status_text.success(f"Document '{uploaded_file.name}' processed successfully!")
                            progress_bar.progress(1.0)
                            time.sleep(2)
                            status_text.empty()
                            progress_bar.empty()
                            break
                        else:
                            status_text.info(f"Processing: {result.get('message', 'In progress...')}")
                    
                    progress = min(elapsed_time / max_wait_time, 0.9)
                    progress_bar.progress(progress)
                    
                except requests.RequestException:
                    # Continue polling if request fails
                    pass
                
                time.sleep(poll_interval)
                elapsed_time += poll_interval
            
            if not st.session_state.current_document:
                st.error("Document processing timed out. Please try again.")
                
        except Exception as e:
            st.error(f"Error processing document: {e}")
        
        finally:
            st.session_state.processing = False

if __name__ == "__main__":
    main()