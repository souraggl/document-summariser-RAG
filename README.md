# PDF Summarizer & Chat with Gemini

This is a Streamlit application that allows users to upload a PDF document and chat with it (ask questions, get summaries) using Google's Gemini Pro model.

## Features
- **Upload PDF**: Easily upload any PDF document.
- **RAG Architecture**: Uses Retrieval Augmented Generation to answer questions based *only* on the document content.
- **Gemini Powered**: Leverages the free tier of Google Gemini API.
- **Chat History**: Maintains conversation context for follow-up questions.

## Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd Summariser
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Get a Gemini API Key:**
    - Visit [Google AI Studio](https://aistudio.google.com/app/apikey) to get your free API key.

4.  **Run the App:**
    ```bash
    streamlit run app.py
    ```

## Usage
- Enter your API Key in the sidebar.
- Upload a PDF file.
- Start chatting!
