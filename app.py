import streamlit as st
from utils.pdf_utils import process_pdf
from utils.text_utils import process_text
from utils.web_utils import process_url
from utils.yt_utils import process_youtube
from utils.rag_pipeline import initialize_rag, answer_query, summarize_doc
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key=os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

st.set_page_config(page_title="Ultimate Summarizer & QA App", layout="wide")
st.title("🧠 Ultimate Summarizer and Query App")
docs = None

source = st.radio("Choose your input source:", ["PDF", "Text", "YouTube", "Website"])

if source == "PDF":
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file:
        docs = process_pdf(uploaded_file)

elif source == "Text":
    uploaded_file = st.file_uploader("Upload a PDF file", type="txt")
    if uploaded_file:
        docs = process_text(uploaded_file)

elif source == "YouTube":
    yt_url = st.text_input("Enter YouTube URL:")
    if yt_url:
        docs = process_youtube(yt_url)

elif source == "Website":
    web_url = st.text_input("Enter Website URL:")
    if web_url:
        docs = process_url(web_url)

if docs:
    rag = initialize_rag(docs)

    if st.button("🔍 Summarize Document"):
        summary = summarize_doc(rag,llm)
        st.subheader("Summary")
        st.write(summary)

    query = st.text_input("Ask a question about the document:")
    if query:
        response = answer_query(rag, query,llm)
        st.subheader("Answer")
        st.write(response)
