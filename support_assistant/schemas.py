"""
schemas.py
Pydantic models for the FastAPI request/response contract.
"""

from typing import List
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
    confidence: float