from fastapi import FastAPI
from fastapi.responses import JSONResponse
from .data_models import (
    LoginRequest,
    LoginResponse,
    BookRecommendation,
    ClickRequest,
    BookDetails
)
from .queries import (
    check_user_exists,
    get_top_recommendations,
    get_global_top_recommendations,
    log_ctr_event,
    get_book_details,
    get_books_read_by_user,
    get_books_not_read_by_user
)

app = FastAPI(title="Goodreads Recommendation API")


# ------------------------------------------------------
# HEALTH CHECK (no external dependencies)
# ------------------------------------------------------
@app.get("/health")
def health_check():
    return {"status": "healthy"}


# ------------------------------------------------------
# 1. LOGIN
# ------------------------------------------------------
@app.post("/load-recommmendation", response_model=LoginResponse)
def login(request: LoginRequest):
    user_id = request.user_id

    # CASE 1: EXISTING USER → personalized recommendations
    if check_user_exists(user_id):
        df = get_top_recommendations(user_id)

        recommendations = [
            BookRecommendation(
                book_id=row["book_id"],
                title=row["title"],
                author=row["author"],
                predicted_rating=row["predicted_rating"]
            )
            for row in df
        ]

        return LoginResponse(user_id=user_id, recommendations=recommendations)

    # CASE 2: NEW USER → global top 10
    df = get_global_top_recommendations()

    recommendations = [
        BookRecommendation(
            book_id=row["book_id"],
            title=row["title"],
            author=row["author"],
            predicted_rating=row["predicted_rating"]
        )
        for row in df
    ]

    return LoginResponse(user_id=user_id, recommendations=recommendations)


# ------------------------------------------------------
# 2. BOOK CLICK - CTR EVENT
# ------------------------------------------------------
@app.post("/book-click", response_model=BookDetails)
def book_click(request: ClickRequest):
    user_id = request.user_id
    book_id = request.book_id

    # Log CTR
    log_ctr_event(user_id, book_id)

    # Get book details
    details = get_book_details(book_id)
    if not details:
        return JSONResponse(status_code=404, content={"message": "book not found"})

    return BookDetails(**details)


# ------------------------------------------------------
# 3. USER HAS READ (is_read = TRUE)
# ------------------------------------------------------
@app.get("/books-read/{user_id}")
def books_read(user_id: str):
    rows = get_books_read_by_user(user_id)
    return {"user_id": user_id, "books_read": rows}


# ------------------------------------------------------
# 4. USER HAS NOT READ
# ------------------------------------------------------
@app.get("/books-unread/{user_id}")
def books_unread(user_id: str):
    rows = get_books_not_read_by_user(user_id)
    return {"user_id": user_id, "books_unread": rows}
