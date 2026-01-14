import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAI  # optional, can replace with local LLM

st.title("Document Summarizer & Q&A Bot (Hugging Face Embeddings)")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Save the PDF temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(documents)

    # -------------------------
    # Create Hugging Face embeddings
    # -------------------------
    # This uses a pre-trained sentence-transformers model
    hf_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model=hf_model)

    # Create vector store (FAISS)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # -------------------------
    # Create LLM
    # -------------------------
    # You can still use OpenAI LLM if you want
    llm = OpenAI(
        temperature=0,
        model="gpt-4o-mini",
        api_key=st.secrets.get("OPENAI_API_KEY")  # optional; can be local LLM
    )

    # Create Retrieval QA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )

    # User question
    question = st.text_input("Ask a question about the document")

    if question:
        answer = qa.run(question)
        st.success(answer)

