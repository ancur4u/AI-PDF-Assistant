from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.llms.ollama import Ollama  # ✅ fallback import

def answer_query(query, vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    docs = retriever.get_relevant_documents(query)

    prompt_template = """
    Use the context below to answer the question:
    ---------------------
    {context}
    ---------------------
    Question: {question}
    Answer:
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    llm = Ollama(model="llama3")
    chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)
    response = chain.run(input_documents=docs, question=query)
    return response, docs