import streamlit as st
import os
import sys
import uuid
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


try:
    from document_processing.loader import PDFLoader
    from document_processing.chunking import DocumentChunker
    from vectorstore.chroma import VectorStoreManager
    from core.orchestrator import Orchestrator
    from core.config import Config
    from audit.logger import AuditLogger
except ImportError as e:
    st.error(f"Import Error: {e}. Please ensure you are running from the correct directory.")
    st.stop()


st.set_page_config(page_title="DocuMind AI", layout="wide", page_icon="📄")


if "messages" not in st.session_state:
    st.session_state.messages = []


ORCHESTRATOR_VERSION = 2  

if "orchestrator" not in st.session_state or st.session_state.get("orchestrator_version") != ORCHESTRATOR_VERSION:
    try:
        st.session_state.vector_store = VectorStoreManager()
        st.session_state.orchestrator = Orchestrator()
        st.session_state.audit_logger = AuditLogger()
        st.session_state.loader = PDFLoader()
        st.session_state.chunker = DocumentChunker()
        st.session_state.orchestrator_version = ORCHESTRATOR_VERSION
        logger.info(f"Orchestrator initialized (version {ORCHESTRATOR_VERSION})")
    except Exception as e:
        st.error(f"Failed to initialize components: {e}")

st.title("📄 DocuMind AI")
st.markdown("### Intelligent Document Analysis Platform")


tab_chat, tab_audit = st.tabs(["💬 Chat", "📊 Audit Logs"])


with st.sidebar:
    st.header("🗂️ Knowledge Base")
    st.text(f"Model: {Config.MODEL_NAME}")
    
    if st.button("🔄 Reset System", help="Clear memory and reload components"):
        st.session_state.clear()
        st.rerun()
    
    if st.button("🗑️ Clear Database", help="Delete all documents from vector store", type="secondary"):
        try:
            st.session_state.vector_store.clear_database()
            st.success("Database cleared successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to clear database: {e}")

    uploaded_files = st.file_uploader("Upload PDFs", type=['pdf'], accept_multiple_files=True)
    
    if st.button("Process Documents", type="primary"):
        if not uploaded_files:
            st.warning("Please upload files first.")
        else:
            with st.status("Processing documents...", expanded=True) as status:
                all_chunks = []
                temp_dir = "data/documents"
                os.makedirs(temp_dir, exist_ok=True)
                
                for uploaded_file in uploaded_files:
                    st.write(f"Processing {uploaded_file.name}...")
                    
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    try:
                        pages = st.session_state.loader.load(file_path)
                        st.write(f"✅ Loaded {len(pages)} pages from {uploaded_file.name}")
                        
                        doc_id = str(uuid.uuid4())
                        chunks = st.session_state.chunker.split_documents(pages, doc_id)
                        for c in chunks:
                            c['metadata']['source'] = uploaded_file.name
                            
                        all_chunks.extend(chunks)
                        
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")
                
                if all_chunks:
                    st.write(f"Embedding {len(all_chunks)} chunks...")
                    st.session_state.vector_store.add_chunks(all_chunks)
                    status.update(label="Processing Complete!", state="complete", expanded=False)
                    st.success(f"Successfully processed {len(all_chunks)} chunks into Knowledge Base.")
                else:
                    status.update(label="Processing Failed", state="error")



with tab_chat:
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    with chat_container:
        if "last_sources" in st.session_state and st.session_state.last_sources:
            with st.expander("📋 View Sources & Reasoning"):
                st.json(st.session_state.last_sources.get("audit_log", []))
                
                docs = st.session_state.last_sources.get("retrieved_docs", [])
                if docs:
                    st.write("**Retrieved Documents:**")
                    for i, doc in enumerate(docs):
                        filename = doc['metadata'].get('source', f'Document {i+1}')
                        page = doc['metadata'].get('page_number', '?')
                        st.caption(f"{filename} - Page {page}")
                        st.text(doc.get('content', '')[:500] + "...")

    prompt = st.chat_input("Ask a question about your documents...")
    
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
        
        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    try:
                        query_id = str(uuid.uuid4())
                        result_state = st.session_state.orchestrator.run(
                            prompt, 
                            query_id=query_id, 
                            audit_logger=st.session_state.audit_logger
                        )
                        
                        response = result_state.get("final_response") or "I couldn't generate an answer. Please check the Audit Logs or ensure Ollama is running."
                        
                        st.session_state.audit_logger.log_query(query_id, result_state)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        st.session_state.last_sources = {
                            "audit_log": result_state.get("audit_log", []),
                            "retrieved_docs": result_state.get("retrieved_docs", [])
                        }
                        
                    except Exception as e:
                        error_msg = f"An error occurred: {e}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})






with tab_audit:
    st.header("System Audit Trail")
    if st.button("Refresh Logs"):
        logs = st.session_state.audit_logger.get_logs()
        st.dataframe(logs)
