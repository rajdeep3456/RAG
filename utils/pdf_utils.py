import tempfile
from langchain_community.document_loaders import PyPDFLoader

def process_pdf(uploaded_file):
    # Create a temporary file and write the uploaded content to it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Pass the path to PyPDFLoader
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    return docs
