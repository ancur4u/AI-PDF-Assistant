from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain_ollama import OllamaLLM

def generate_summary(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    llm = OllamaLLM(model="llama3")
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.invoke(docs)
    return summary