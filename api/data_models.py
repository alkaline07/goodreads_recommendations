from pydantic import BaseModel
from typing import List
from typing import Optional, Literal


class LoginRequest(BaseModel):
    user_id: str


class BookRecommendation(BaseModel):
    book_id: int
    predicted_rating: float
    title: str
    author: str


class LoginResponse(BaseModel):
    user_id: str
    recommendations: List[BookRecommendation]


class ClickRequest(BaseModel):
    user_id: str
    book_id: int
    event_type: str
    book_title: Optional[str] = None


class BookDetails(BaseModel):
    book_id: int
    title: str
    description: str
    authors: List[str]
    average_rating: float
    ratings_count: int

class MarkReadRequest(BaseModel):
    user_id: str
    book_id: int
    rating: Optional[int] = 0


