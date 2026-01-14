import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, OpenAI


st.title("Document Summarizer & Q&A Bot")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Save uploaded file
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

   st.write("API key loaded:", bool(st.secrets.get("OPENAI_API_KEY")))
# Create embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=st.secrets["OPENAI_API_KEY"]
    )

    # Create vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Create LLM
    llm = OpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=st.secrets["OPENAI_API_KEY"]
    )

    # Create QA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )

    # Ask question
    question = st.text_input("Ask a question about the document")

    if question:
        answer = qa.run(question)
        st.success(answer)
