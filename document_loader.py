from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os


def load_and_chunk_docs(folder_path):
    docs = []
    for file in os.listdir(folder_path):
        if file.endswith(".docx"):
            path = os.path.join(folder_path, file)
            loader = UnstructuredWordDocumentLoader(path)
            docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(docs)
