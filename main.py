from document_loader import load_and_chunk_docs
from embedder import Embedder
from generator import Generator
from config import DOCS_FOLDER, EMBEDDING_MODEL_NAME, LLM_MODEL_NAME

docs = load_and_chunk_docs(DOCS_FOLDER)

embedder = Embedder(EMBEDDING_MODEL_NAME)
embedder.build_vectorstore(docs)

generator = Generator(LLM_MODEL_NAME)


def rag_answer(query):
    relevant_docs = embedder.retrieve(query, k=5)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    return generator.generate(context, query)


# Example
if __name__ == "__main__":
    query = "What is discussed about Artificial Intelligence (AI)?"
    res = rag_answer(query)
    print(res)
