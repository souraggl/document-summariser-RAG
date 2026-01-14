import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, OpenAI

st.title("Document Summarizer & Q&A Bot")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Save the uploaded PDF temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load PDF into documents
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(documents)

    # --- CREATE EMBEDDINGS ---
    embeddings = OpenAIEmbeddings(
        api_key=st.secrets["OPENAI_API_KEY"],  # Make sure this key is correct in Streamlit secrets
        model="text-embedding-3-small"        # Use an active embedding model
    )

    # --- CREATE VECTOR STORE ---
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # --- CREATE LLM ---
    llm = OpenAI(
        temperature=0,
        api_key=st.secrets["OPENAI_API_KEY"],
        model="gpt-4o-mini"   # Recommended model, works for Q&A
    )

    # --- CREATE QA CHAIN ---
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )

    # User input question
    question = st.text_input("Ask a question about the document")

    if question:
        answer = qa.run(question)
        st.success(answer)

