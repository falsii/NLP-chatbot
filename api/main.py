import os
import sys
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)


from chat import load_chatbot, predict_chatbot
from database import init_database, save_chat_log, save_feedback, get_recent_logs, get_feedback_summary


app = FastAPI(
    title="Offline PyTorch NLP Chatbot API",
    description="FastAPI backend for an offline chatbot using PyTorch, TF-IDF retrieval, rule-based safety, and SQLite logging.",
    version="1.1.0"
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=500)
    include_debug: bool = True


class ChatResponse(BaseModel):
    reply: str
    log_id: Optional[int] = None
    intent: Optional[str] = None
    confidence: Optional[float] = None
    source: Optional[str] = None
    faq_match_score: Optional[float] = None
    faq_question: Optional[str] = None


class FeedbackRequest(BaseModel):
    log_id: int
    feedback: str = Field(..., description="Allowed values: helpful, not_helpful")
    feedback_comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    success: bool
    message: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    database_ready: bool
    message: str


chatbot_resources = {
    "model": None,
    "intents": None,
    "all_words": None,
    "tags": None,
    "device": None,
    "retriever": None,
    "loaded": False
}


@app.on_event("startup")
def startup_event():
    try:
        init_database()

        model, intents, all_words, tags, device, retriever = load_chatbot()

        chatbot_resources["model"] = model
        chatbot_resources["intents"] = intents
        chatbot_resources["all_words"] = all_words
        chatbot_resources["tags"] = tags
        chatbot_resources["device"] = device
        chatbot_resources["retriever"] = retriever
        chatbot_resources["loaded"] = True

        print("Chatbot resources loaded successfully.")
        print("Database initialized successfully.")

    except Exception as error:
        chatbot_resources["loaded"] = False
        print(f"Startup failed: {error}")


@app.get("/", response_model=HealthResponse)
def root():
    return {
        "status": "ok",
        "model_loaded": chatbot_resources["loaded"],
        "database_ready": True,
        "message": "Offline PyTorch NLP Chatbot API is running."
    }


@app.get("/health", response_model=HealthResponse)
def health_check():
    return {
        "status": "ok" if chatbot_resources["loaded"] else "error",
        "model_loaded": chatbot_resources["loaded"],
        "database_ready": True,
        "message": "Model and database are ready." if chatbot_resources["loaded"] else "Model is not loaded."
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not chatbot_resources["loaded"]:
        raise HTTPException(
            status_code=503,
            detail="Chatbot model is not loaded. Please train the model first using: python src/train.py"
        )

    user_message = request.message.strip()

    result = predict_chatbot(
        user_message=user_message,
        model=chatbot_resources["model"],
        intents=chatbot_resources["intents"],
        all_words=chatbot_resources["all_words"],
        tags=chatbot_resources["tags"],
        device=chatbot_resources["device"],
        retriever=chatbot_resources["retriever"]
    )

    log_id = save_chat_log(
        user_message=user_message,
        bot_reply=result["response"],
        intent=result.get("intent"),
        confidence=result.get("confidence"),
        source=result.get("source"),
        faq_match_score=result.get("faq_match_score"),
        faq_question=result.get("faq_question")
    )

    if request.include_debug:
        return {
            "reply": result["response"],
            "log_id": log_id,
            "intent": result.get("intent"),
            "confidence": result.get("confidence"),
            "source": result.get("source"),
            "faq_match_score": result.get("faq_match_score"),
            "faq_question": result.get("faq_question")
        }

    return {
        "reply": result["response"],
        "log_id": log_id
    }


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(request: FeedbackRequest):
    try:
        updated = save_feedback(
            log_id=request.log_id,
            feedback=request.feedback,
            feedback_comment=request.feedback_comment
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    if not updated:
        raise HTTPException(status_code=404, detail="Chat log not found.")

    return {
        "success": True,
        "message": "Feedback saved successfully."
    }


@app.get("/logs")
def logs(limit: int = 50):
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="Limit must be between 1 and 200.")

    return {
        "logs": get_recent_logs(limit=limit)
    }


@app.get("/feedback-summary")
def feedback_summary():
    return get_feedback_summary()