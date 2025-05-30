import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA, LLMChain, AnalyzeDocumentChain, load_summarize_chain
from langchain.callbacks import get_openai_callback
from langchain import PromptTemplate
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings



def initialize_rag(docs: list[Document], chunk_size: int = 2000, chunk_overlap: int = 200):
    """
    Initializes RAG by chunking documents and creating a FAISS vector store.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    chunked_docs = []
    for doc in docs:
        chunks = splitter.split_text(doc.page_content)
        for chunk in chunks:
            chunked_docs.append(Document(page_content=chunk, metadata=doc.metadata))

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = Chroma.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        persist_directory="chroma_db",
        collection_name="my_collection" 
    )

    return {
        'vector_store': vector_store,
        'embeddings': embeddings,
        'chunks': chunked_docs
    }

def summarize_doc(rag: dict,llm) -> str:
    docs=rag['chunks']
    llm=llm
    chunks_prompt="""
    Please summarize the below document:
    document:`{text}'
    Summary:
    """
    map_prompt_template=PromptTemplate(input_variables=['text'],
                                        template=chunks_prompt)
    
    final_prompt='''
    Provide the final summary of the entire document with these important points.
    Add a Title,Start the precise summary with an introduction and provide the summary in number 
    points for the document.
    document:{text}

    '''
    final_prompt_template=PromptTemplate(input_variables=['text'],template=final_prompt)

    summary_chain=load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=map_prompt_template,
        combine_prompt=final_prompt_template,
        verbose=True
    )

    output=summary_chain.run(docs)

    return output

def answer_query(rag: dict, query: str, llm) -> str:
    """
    Retrieve relevant document chunks and generate an answer to the query.
    """
    vector_store = rag['vector_store']
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    llm = llm
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    with get_openai_callback() as cb:
        result = qa_chain.run(query)
        print(f"Tokens used: {cb.total_tokens}")

    return result