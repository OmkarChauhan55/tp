from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import pdfplumber
import tempfile
import os
import requests

from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 🚀 FastAPI app instance
main = FastAPI(
    title="HackRx Retrieval System",
    version="1.0.0",
    description="LLM-powered API to retrieve answers from insurance policies using LangChain & Gemini."
)

# 🌐 CORS settings
main.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔐 API keys
GOOGLE_API_KEY = "AIzaSyBzFr-G4_pZG_lxDrMDO1O3-n4WIkKHUUQ"
TEAM_TOKEN = "ee780205a54c3c1504fd981ed73efa751d8b9a453087a3f5a9b9d03c8e93ed83"
security = HTTPBearer()

# ✅ Token Validator
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid auth scheme")
    if credentials.credentials != TEAM_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return True

# 📦 Request schema
class AskRequest(BaseModel):
    documents: str
    questions: list[str]

# 📄 PDF loader function
def load_pdf_from_url(pdf_url: str) -> str:
    response = requests.get(pdf_url)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to download PDF")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(response.content)
        temp_path = temp_file.name

    full_text = ""
    with pdfplumber.open(temp_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

    os.unlink(temp_path)
    return full_text

# ⚡ Cache vectorstore to avoid repeated work
@lru_cache(maxsize=10)
def get_cached_vectorstore(doc_url: str):
    full_text = load_pdf_from_url(doc_url)
    chunks = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000).split_text(full_text)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    return FAISS.from_texts(chunks, embedding=embeddings)

# 🧠 Q&A endpoint
@main.post("/api/v1/hackrx/run", tags=["HackRx"])
async def run_hackrx(body: AskRequest, _: bool = Depends(verify_token)):
    # ⚡ Get cached vectorstore
    vector_store = get_cached_vectorstore(body.documents)

    # 📝 Prompt Template (Strict and Concise)
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are an expert assistant.

Answer the following question using ONLY the context below.
Respond word-to-word from the policy where possible. Do NOT add extra explanation.
If not found, just say: "Answer not available in the context."

Context:
{context}

Question: {question}
Strict Answer:"""
    )

    # 🤖 Load Gemini model
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2,
        google_api_key=GOOGLE_API_KEY
    )

    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    # 🔁 Process all questions
    answers = []
    for question in body.questions:
        docs = vector_store.similarity_search(question, k=1)  # Focused context
        result = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
        answers.append(result["output_text"].strip())  # Clean output

    return {"answers": answers}
