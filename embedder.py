from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS


class Embedder:
    def __init__(self, model_name):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.vectorstore = None

    def build_vectorstore(self, docs):
        self.vectorstore = FAISS.from_documents(docs, self.embeddings)

    def retrieve(self, query, k=5):
        return self.vectorstore.similarity_search(query, k=k)
