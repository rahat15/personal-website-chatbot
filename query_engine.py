from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser, TokenTextSplitter
from llama_index.core.schema import Document
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

def load_query_engine():
    # Step 1: Load raw markdown-style documents
    documents = SimpleDirectoryReader("data").load_data()

    # Step 2: Use MarkdownNodeParser to chunk by headings
    markdown_parser = MarkdownNodeParser()
    structured_nodes = markdown_parser.get_nodes_from_documents(documents)

    # Step 3: Convert nodes to pseudo-documents
    structured_docs = [Document(text=node.text) for node in structured_nodes]

    # Step 4: Token-based chunking on those documents
    splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=64)
    nodes = splitter.get_nodes_from_documents(structured_docs)

    # Step 5: Debug output
    with open("debug_chunks.txt", "w", encoding="utf-8") as f:
        f.write(f"✅ Total Chunks Created: {len(nodes)}\n\n")
        for i, node in enumerate(nodes):
            f.write(f"--- Chunk #{i+1} ---\n")
            f.write(node.text.strip() + "\n\n")

    # Step 6: Embeddings
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Step 7: Groq LLM
    Settings.llm = Groq(
        model="llama3-8b-8192",
        api_key=os.getenv("GROQ_API_KEY")
    )

    # Step 8: Create index + query engine
    index = VectorStoreIndex(nodes)
    query_engine = index.as_query_engine(similarity_top_k=5)

    return query_engine

query_engine = load_query_engine()
