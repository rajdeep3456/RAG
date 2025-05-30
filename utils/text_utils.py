from langchain_community.document_loaders import TextLoader

def process_text(uploaded_file):
    loader=TextLoader(uploaded_file)
    docs=loader.load()
    docs