from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import pdfplumber
import tempfile
import os
import requests

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 🚀 FastAPI app instance
main = FastAPI(
    title="HackRx Retrieval System",
    version="1.0.0",
    description="LLM-powered API to retrieve short, accurate answers from insurance policies."
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

# 🧠 Q&A endpoint
@main.post("/api/v1/hackrx/run", tags=["HackRx"])
async def run_hackrx(body: AskRequest, _: bool = Depends(verify_token)):
    # 📖 Extract text from PDF
    full_text = load_pdf_from_url(body.documents)

    # 📚 Split text into small chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(full_text)

    # 🔍 Vectorstore creation
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)

    # 📝 Concise Prompt Template
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""Answer briefly using the context below.
If not found, reply: "Not available."

Context:
{context}

Question: {question}
Short Answer:"""
    )

    # 🤖 Gemini model
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY
    )

    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    # 🔁 Process all questions
    answers = []
    for question in body.questions:
        docs = vector_store.similarity_search(question, k=2)
        result = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
        answers.append(result["output_text"].strip())

    return {"answers": answers}
