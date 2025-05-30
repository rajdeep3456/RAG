from langchain_community.document_loaders import WebBaseLoader
import bs4

def process_url(url):
    loader=WebBaseLoader(web_paths=[url])
    docs=loader.load()
    return docs