from fastapi import FastAPI
from pydantic import BaseModel
from query_engine import query_engine
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

@app.post("/chat")
def ask(req: QuestionRequest):
    response = query_engine.query(req.question)
    return {"answer": response.response}
