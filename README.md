# Local RAG System with LLaMA 3.2 11B and Word Documents
This project demonstrates a Retrieval-Augmented Generation (RAG) system that uses:
* Local Word .docx documents as a knowledge base
* LangChain for document loading, chunking, embedding, and retrieval
* FAISS for vector similarity search
* Meta’s LLaMA 3.2 11B Chat model (via Hugging Face) for text generation

## Setup Instructions
1. Clone or Download the Repository

2. Install Dependencies

Make sure you have Python 3.9 or higher installed. Then install the requirements:

```
pip install -r requirements.txt
```

The requirements do not include pytorch because setup can be specific to your machine. Install pytorch per the instructions here: https://pytorch.org/get-started/locally/

3. Hugging Face Login and Model Access
Login to Hugging Face CLI:

```
huggingface-cli login
```

Visit the following URL to accept Meta’s license agreement for the LLaMA 3.2 11B Chat model:

https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct

4. Prepare Your Word Documents
Place your .docx files into the ./word_docs directory.

Example:

rag_project/  
├── word_docs/  
│ ├── doc1.docx  
│ ├── doc2.docx  
│ └── ...  

5. Run the RAG System

```
python main.py
```

The script will load and chunk documents, build embeddings, retrieve relevant chunks, and generate answers using the LLaMA 3.2 11B model

## Project Structure
main.py: Entry point demonstrating example usage

document_loader.py: Loads and chunks Word documents with LangChain

embedder.py: Embeds documents and performs vector similarity search using FAISS

generator.py: Loads and wraps the LLaMA Hugging Face model for generation

config.py: Central configuration for paths and model names

## Customization
Modify chunk sizes and overlaps in document_loader.py with RecursiveCharacterTextSplitter

Change embedding or LLM model in config.py to swap models easily

Extend main.py to accept dynamic queries or build a web interface

## Notes
Requires an NVIDIA GPU with sufficient VRAM (like RTX 4090) to run the model efficiently locally

Ensure you comply with Meta’s license terms when using the LLaMA 3.2 model

## License
This project uses Meta’s LLaMA 3.2 model under Meta’s licensing terms. Please review the license before usage
