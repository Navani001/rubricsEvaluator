"""
Data models for PDF and Quizz requests
"""
from typing import Optional
from pydantic import BaseModel


class PDFRequest(BaseModel):
    """Request model for PDF queries"""
    query: str
    book: Optional[str] = None
    message: list[dict] = []
    n_results: Optional[int] = 3


class QuizzRequest(BaseModel):
    """Request model for quiz generation"""
    book: Optional[str] = None
    n_results: Optional[int] = 3
    question: Optional[str] = 10


class RubricGenRequest(BaseModel):
    """Request model for rubric generation"""
    text: str
    topic: Optional[str] = None
