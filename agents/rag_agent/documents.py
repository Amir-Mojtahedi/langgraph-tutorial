import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

file_path = "agents/rag_agent/sample_data/nke-10k-2023.pdf"
py_pdf_loader = PyPDFLoader(file_path)
docs: list[Document] = py_pdf_loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)

all_splits: list[Document] = text_splitter.split_documents(docs)

embeddings = OllamaEmbeddings(
    model=os.environ["OLLAMA_EMBEDDING_MODEL"],
    base_url=os.environ["OLLAMA_BASE_URL"],
)

vector_1 = embeddings.embed_query(text=all_splits[0].page_content)
vector_2 = embeddings.embed_query(text=all_splits[1].page_content)

assert len(vector_1) == len(vector_2)
print(f"Generated vectors of length {len(vector_1)}\n")
print(vector_1[:10])

vector_store = InMemoryVectorStore(embedding=embeddings)

ids = vector_store.add_documents(documents=all_splits)

results = vector_store.similarity_search(
    query="How many distribution centers does Nike have in the US?"
)

print(results[0])
