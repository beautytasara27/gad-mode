import os
import requests
from flask import Flask, request, jsonify
from typing import Optional
from bs4 import BeautifulSoup

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
from serpapi import GoogleSearch
from dotenv import load_dotenv
from flask_cors import CORS


load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

app = Flask(__name__)
CORS(app)


class ChatOpenRouter(ChatOpenAI):
    def __init__(self, openai_api_key: Optional[str] = None, **kwargs):
        openai_api_key = openai_api_key or OPENROUTER_API_KEY
        super().__init__(
            base_url="https://openrouter.ai/api/v1",
            openai_api_key=openai_api_key,
            **kwargs
        )


def load_web_pages(urls):
    documents = []
    for url in urls:
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            documents.extend(docs)
            print(f"Loaded web content from {url}")
        except Exception as e:
            print(f"Error loading {url}: {e}")
    return documents

def load_documents(file_paths):
    documents = []
    for path in file_paths:
        try:
            loader = PyPDFLoader(path) if path.endswith(".pdf") else TextLoader(path)
            docs = loader.load()
            documents.extend(docs)
            print(f"Loaded local document: {path}")
        except Exception as e:
            print(f"Error loading {path}: {e}")
    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)

def create_llm_chain():
    prompt = ChatPromptTemplate.from_template("""
    You are an admissions officer responding to emails on behalf of the Graduate Admissions Office. Use the context below to craft a professional, helpful, and human-sounding email response.

    Respond as if you are writing a real email to the student. Be clear and polite. Avoid robotic or overly formal phrasing such as "the information available is." If you don’t know the answer, say so politely and suggest who the student can contact or where they can look for further help.

    Context:
    {context}

    Email from Student:
    {question}

    Your Email Response:
    """)
    llm = ChatOpenRouter(
        model_name="anthropic/claude-3.7-sonnet:thinking",
        temperature=0.0,
        streaming=False,
        max_tokens=1024
    )
    return prompt | llm

def serpapi_search(query):
    print(f"🔍 Searching for: '{query}' using SerpAPI")
    params = {
        "q": f"site:stevens.edu {query}",
        "api_key": SERPAPI_API_KEY,
        "num": 3,
        "engine": "google"
    }
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        urls = [res["link"] for res in results.get("organic_results", []) if "link" in res]
        if not urls:
            return ""
        texts = []
        for url in urls:
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")
                text = soup.get_text(separator=" ", strip=True)
                texts.append(text[:3000])
            except Exception as e:
                print(f"Could not fetch {url}: {e}")
        return "\n\n".join(texts)
    except Exception as e:
        print(f"SerpAPI error: {e}")
        return ""


def hybrid_answer(query, vectorstore, llm_chain):
    retriever = vectorstore.as_retriever()
    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)

    if len(context) < 100:
        print("Using fallback web search...")
        context = serpapi_search(query)
        if not context.strip():
            return "No relevant information found in documents or online."

    try:
        result = llm_chain.invoke({"context": context, "question": query})
        return result.content
    except Exception as e:
        print(f"LLM failed: {e}")
        return "Error generating a response."


print("--- Initializing RAG System ---")
web_urls = [
    "https://www.stevens.edu/admission-aid/graduate-admissions/graduate-events-and-open-houses",
    "https://www.stevens.edu/isss",
    "https://www.stevens.edu/student-life/student-health-services"
]
file_paths = ["docs/inbox.pdf", "docs/visa.pdf"]

web_docs = load_web_pages(web_urls)
file_docs = load_documents(file_paths)
all_docs = web_docs + file_docs

if not all_docs:
    print("No documents found.")
    vectorstore = None
    llm_chain = None
else:
    chunks = split_documents(all_docs)
    vectorstore = build_vectorstore(chunks)
    llm_chain = create_llm_chain()


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "Missing 'query' in request body"}), 400

    if not vectorstore or not llm_chain:
        return jsonify({"error": "System not initialized properly"}), 500

    answer = hybrid_answer(query, vectorstore, llm_chain)
    return jsonify({"response": answer})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
