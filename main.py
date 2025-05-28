import streamlit as st
import os
import json
import time as t
from datetime import datetime
from PIL import Image
import fitz  # PyMuPDF
from fpdf import FPDF
from packaging import version

from app.pdf_utils import split_pdf, convert_pdf_to_word, convert_pdf_to_pptx, get_split_chunks
from app.rag_engine import answer_query
from app.history_manager import save_to_history, load_history
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms.ollama import Ollama
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfMerger

# Setup directories
os.makedirs("data/uploads", exist_ok=True)
os.makedirs("data/splits", exist_ok=True)
os.makedirs("data/word_docs", exist_ok=True)

UPLOAD_LOG = "data/upload_log.json"

def log_file_upload(filename, summary_time=None, word_count=None):
    os.makedirs("data", exist_ok=True)
    log = []
    if os.path.exists(UPLOAD_LOG):
        with open(UPLOAD_LOG, "r") as f:
            log = json.load(f)
    entry = {"file": filename, "time": datetime.now().isoformat()}
    if summary_time is not None:
        entry["summary_time_sec"] = round(summary_time, 2)
    if word_count is not None:
        entry["word_count"] = word_count
    log.append(entry)
    with open(UPLOAD_LOG, "w") as f:
        json.dump(log, f, indent=2)

def load_upload_history():
    if os.path.exists(UPLOAD_LOG):
        with open(UPLOAD_LOG, "r") as f:
            return json.load(f)
    return []

def render_all_pages(pdf_path, max_pages=20):
    doc = fitz.open(pdf_path)
    images = []
    for i in range(min(len(doc), max_pages)):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append((i + 1, img))
    return images

def export_summary_to_pdf(summary_text, output_path="summary.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in summary_text.split("\n"):
        pdf.multi_cell(0, 10, line)
    pdf.output(output_path)

st.set_page_config(page_title="PDF AI Assistant", layout="wide")
st.title("\U0001F4C4 PDF AI Assistant")

# Sidebar analytics
with st.sidebar.expander("\U0001F4CA App Usage Summary", expanded=False):
    logs = load_upload_history()
    total_files = len(logs)
    total_words = sum(log.get("word_count", 0) for log in logs)
    total_time = sum(log.get("summary_time_sec", 0) for log in logs)
    st.markdown(f"- **Total Files Processed:** {total_files}")
    st.markdown(f"- **Total Words Summarized:** {total_words:,}")
    st.markdown(f"- **Total Summarization Time:** {total_time:.2f} seconds")

# Sidebar file history
with st.sidebar.expander("\U0001F4C1 Upload History", expanded=False):
    history = load_upload_history()
    history_files = [h["file"] for h in reversed(history)]
    selected_history_file = st.selectbox("Reopen a previous document", history_files) if history_files else None

# Sidebar PDF preview control
max_preview = st.sidebar.slider("Pages to Preview", min_value=1, max_value=30, value=5)

# Feature selector
task = st.sidebar.radio("Select a Feature", [
    "Split PDF", 
    "Summarize PDF", 
    "Convert to Word", 
    "Convert to PPTX", 
    "Chat with PDF (RAG)",
    "Merge PDFs"
])

# Upload logic for single-file tasks
if task != "Merge PDFs":
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

    if uploaded_file:
        filename = uploaded_file.name
        pdf_path = os.path.join("data/uploads", filename)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success(f"Document successfully uploaded: {filename}")

    elif selected_history_file:
        filename = selected_history_file
        pdf_path = os.path.join("data/uploads", filename)
        st.success(f"Document loaded from history: {filename}")

    else:
        st.info("Please upload a PDF or select a document from history to continue.")
        st.stop()

    with st.expander("\U0001F4C4 Full Document Preview", expanded=False):
        try:
            pages = render_all_pages(pdf_path, max_pages=max_preview)
            for i, img in pages:
                st.image(img, caption=f"Page {i}", use_column_width=True)
        except Exception as e:
            st.warning(f"Unable to preview document: {e}")


if task == "Split PDF":
    total_pages = split_pdf(pdf_path, "data/splits")
    st.success(f"Document split into {total_pages} pages. Files saved in /data/splits.")

elif task == "Summarize PDF":
    st.info("Loading and analyzing document...")
    max_chunks = st.sidebar.slider("Max Chunks to Summarize", min_value=1, max_value=30, value=10)
    chunks = get_split_chunks(pdf_path)[:max_chunks]

    if not chunks:
        st.error("Unable to extract or split content. Ensure the document is not image-scanned.")
        st.stop()

    st.success(f"{len(chunks)} chunks ready for summarization.")
    st.code(chunks[0].page_content[:500], language="text")

    summary_mode = st.radio("Summary Mode", ["Brief", "Detailed"], horizontal=True)
    cache_key = f"{filename.replace('.pdf', '')}_{summary_mode.lower()}_{max_chunks}"
    summary_cache_file = os.path.join("data", f"summary_cache_{cache_key}.json")

    if os.path.exists(summary_cache_file):
        st.success("Cached summary found.")
        with open(summary_cache_file, "r") as f:
            cached = json.load(f)
            st.subheader("\U0001F4DC Final Summary")
            st.write(cached["summary"])
            with open(cached["export_path"], "rb") as pdf_file:
                st.download_button("Download Summary PDF", pdf_file, file_name=os.path.basename(cached["export_path"]))
        st.stop()

    progress_bar = st.progress(0.0, text="Initializing LLaMA3...")
    try:
        llm = OllamaLLM(model="llama3")
        chunk_summaries = []
        for idx, chunk in enumerate(chunks):
            progress = (idx + 1) / len(chunks)
            progress_bar.progress(progress * 0.85, text=f"Summarizing chunk {idx+1}/{len(chunks)}...")
            prompt = f"""Summarize the following section {'briefly' if summary_mode == 'Brief' else 'in detail'}:\n\n{chunk.page_content.strip()}"""
            response = llm.invoke(prompt)
            summary = response.get("content", response) if isinstance(response, dict) else response
            chunk_summaries.append(f"### Section {idx + 1}\n{summary.strip()}")

        with st.expander("🔍 Section Summaries", expanded=False):
            for summary in chunk_summaries:
                st.markdown(summary)

        progress_bar.progress(0.9, text="Combining section summaries...")
        merge_prompt = f"""
Combine the following {'brief' if summary_mode == 'Brief' else 'detailed'} section-wise summaries into one comprehensive final summary:

{chr(10).join(chunk_summaries)}

Ensure the summary captures all key points and is structured clearly.
"""
        final_response = llm.invoke(merge_prompt)
        final_summary = final_response.get("content", final_response) if isinstance(final_response, dict) else final_response

        st.success("Summary complete.")
        st.subheader("🧾 Final Summary")
        st.write(final_summary)

        export_path = os.path.join("data", f"summary_{cache_key}.pdf")
        export_summary_to_pdf(final_summary, export_path)
        with open(export_path, "rb") as f:
            st.download_button("Download Summary PDF", f, file_name=os.path.basename(export_path))

        with open(summary_cache_file, "w") as f:
            json.dump({"summary": final_summary, "export_path": export_path}, f)

        word_count = len(final_summary.split())
        log_file_upload(filename, summary_time=word_count / 2, word_count=word_count)
        progress_bar.progress(1.0, text="Done.")
    except Exception as e:
        st.error(f"An error occurred during summarization: {e}")

elif task == "Convert to Word":
    word_path = os.path.join("data/word_docs", filename.replace(".pdf", ".docx"))
    convert_pdf_to_word(pdf_path, word_path)
    st.success("Conversion to Word document completed.")
    with open(word_path, "rb") as f:
        st.download_button("Download Word Document", f, file_name=os.path.basename(word_path))

elif task == "Convert to PPTX":
    pptx_path = os.path.join("data", filename.replace(".pdf", ".pptx"))
    try:
        st.info("Converting PDF to PowerPoint...")
        convert_pdf_to_pptx(pdf_path, pptx_path)
        st.success("Conversion completed.")
        with open(pptx_path, "rb") as f:
            st.download_button("Download PPTX File", f, file_name=os.path.basename(pptx_path))
    except Exception as e:
        st.error(f"Error during conversion: {e}")

elif task == "Chat with PDF (RAG)":
    st.subheader("\U0001F4AC Ask Questions About This Document")
    progress_bar = st.progress(0, text="Initializing chat engine...")
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load_and_split()
        splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=64)
        chunks = splitter.split_documents(docs)

        texts = [doc.page_content for doc in chunks]
        metadatas = [doc.metadata for doc in chunks]

        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
        progress_bar.progress(1.0, text="Chat engine ready.")

        query = st.text_input("Enter your question below")
        if query:
            response, docs = answer_query(query, vectorstore)
            save_to_history(query, response)
            st.markdown(f"**Response from Assistant:** {response}")

        history = load_history()
        if history:
            st.markdown("---")
            st.subheader("\U0001F4DC Chat History")
            for entry in history:
                st.markdown(f"**You:** {entry['query']}  \n**Assistant:** {entry['response']}")

    except Exception as e:
        st.error(f"Unable to create vector index: {e}")

elif task == "Merge PDFs":
    st.subheader("🗂️ Merge Multiple PDF Files")
    uploaded_files = st.file_uploader("Upload PDF files to merge (drag & drop supported)", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        st.markdown("### ↕️ Reorder Uploaded PDFs")
        file_names = [f.name for f in uploaded_files]

        # Auto-switching logic for reordering UI
        streamlit_version = version.parse(st.__version__)
        if streamlit_version >= version.parse("1.25.0"):
            reordered_names = st.data_editor(file_names, num_rows="dynamic", key="reorder_pdfs")
        else:
            reordered_names = []
            remaining_files = uploaded_files.copy()
            for i in range(len(uploaded_files)):
                file_options = [f.name for f in remaining_files]
                selected = st.selectbox(f"Select file #{i+1}", file_options, key=f"order_{i}")
                for f in remaining_files:
                    if f.name == selected:
                        reordered_names.append(selected)
                        remaining_files.remove(f)
                        break

        ordered_files = []
        for name in reordered_names:
            for f in uploaded_files:
                if f.name == name:
                    ordered_files.append(f)
                    break

        if st.button("🔀 Merge PDFs Now"):
            file_paths = []
            for file in ordered_files:
                file_path = os.path.join("data/uploads", file.name)
                with open(file_path, "wb") as f_out:
                    f_out.write(file.read())
                file_paths.append(file_path)

            merged_output_path = os.path.join("data", "merged.pdf")
            merger = PdfMerger()
            for path in file_paths:
                merger.append(path)
            merger.write(merged_output_path)
            merger.close()

            st.success("✅ PDFs merged successfully!")
            try:
                st.markdown("### 👀 Preview of Merged PDF (First 5 Pages)")
                doc = fitz.open(merged_output_path)
                for i in range(min(len(doc), 5)):
                    page = doc.load_page(i)
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    st.image(img, caption=f"Page {i + 1}", use_column_width=True)
            except Exception as e:
                st.warning(f"Unable to preview merged PDF: {e}")

            with open(merged_output_path, "rb") as f:
                st.download_button("📥 Download Merged PDF", f, file_name="merged.pdf")
