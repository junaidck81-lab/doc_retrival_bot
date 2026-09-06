import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Document QA Bot", layout="centered")
st.title("Document Retrieval & QA Bot 🤖")

# Fetch API key securely from Streamlit Secrets or Environment Variables
groq_api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))

if not groq_api_key:
    st.error("GROQ_API_KEY is not configured. Please add it to Streamlit Secrets.")
    st.stop()

# Initialize Models
@st.cache_resource
def get_vector_tools():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    llm = ChatGroq(
        temperature=0.2,
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile"
    )
    return embeddings, llm

embeddings, llm = get_vector_tools()

os.makedirs("uploads", exist_ok=True)

# File Uploader
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file:
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Read and split document
    with st.spinner("Processing document..."):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)
        
        # Store in vector database
        vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    st.success(f"Processed: {uploaded_file.name}")

    # Question Answering
    question = st.text_input("Ask anything from the uploaded document:")
    
    if question:
        template = """Use only the following context to answer the question. 
If the answer cannot be found in the context, say "The answer is not present in the document."

Context:
{context}

Question: {question}

Helpful Answer:"""
        prompt = PromptTemplate.from_template(template)

        def format_docs(retrieved_docs):
            return "\n\n".join(d.page_content for d in retrieved_docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        with st.spinner("Analyzing document..."):
            answer = rag_chain.invoke(question)

        st.markdown("### Answer")
        st.write(answer)
        st.divider()

        # File Retrieval
        with open(file_path, "rb") as f:
            st.download_button(
                label="📥 Retrieve / Download Original File",
                data=f,
                file_name=uploaded_file.name,
                mime="application/pdf"
            )