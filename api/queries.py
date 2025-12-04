from google.cloud import bigquery
from .database import get_bq_client

_client = None
_project = None
dataset = "books"


def _get_client():
    global _client, _project
    if _client is None:
        _client = get_bq_client()
        _project = _client.project
    return _client, _project


# ---------------------------------------------------------------------
# CHECK IF USER EXISTS IN PREDICTION TABLE
# ---------------------------------------------------------------------
def check_user_exists(user_id: str) -> bool:
    client, project = _get_client()
    query = f"""
    SELECT COUNT(*) AS cnt
    FROM `{project}.{dataset}.boosted_tree_rating_predictions`
    WHERE user_id_clean = '{user_id}'
    """
    job = client.query(query)
    row = next(job.result())
    return row["cnt"] > 0


# ---------------------------------------------------------------------
# GET TOP 10 RECOMMENDATIONS FOR EXISTING USER
# ---------------------------------------------------------------------
def get_top_recommendations(user_id: str):
    client, project = _get_client()
    query = f"""
    SELECT
        pred.book_id,
        pred.predicted_rating,
        b.title_clean AS title,
        IFNULL(b.authors_flat[OFFSET(0)], "Unknown") AS author
    FROM `{project}.{dataset}.boosted_tree_rating_predictions` AS pred
    LEFT JOIN `{project}.{dataset}.goodreads_books_cleaned` AS b
        ON pred.book_id = b.book_id
    WHERE pred.user_id_clean = '{user_id}'
    ORDER BY pred.predicted_rating DESC
    LIMIT 10
    """
    job = client.query(query)
    return [dict(row) for row in job.result()]


# ---------------------------------------------------------------------
# GET GLOBAL TOP 10 BOOKS FOR NEW USERS
# ---------------------------------------------------------------------
def get_global_top_recommendations():
    client, project = _get_client()
    query = f"""
    SELECT
        book_id,
        average_rating AS predicted_rating,
        title_clean AS title,
        IFNULL(author, "Unknown") AS author
    FROM `{project}.{dataset}.global_top10_books`
    ORDER BY predicted_rating DESC
    """
    job = client.query(query)
    return [dict(row) for row in job.result()]


# ---------------------------------------------------------------------
# LOG CTR EVENT
# ---------------------------------------------------------------------
def log_ctr_event(user_id: str, book_id: int):
    client, project = _get_client()
    table_id = f"{project}.{dataset}.user_ctr_events"
    rows = [{"user_id": user_id, "book_id": book_id}]
    errors = client.insert_rows_json(table_id, rows)
    return len(errors) == 0


# ---------------------------------------------------------------------
# BOOK DETAILS
# ---------------------------------------------------------------------
def get_book_details(book_id: int):
    client, project = _get_client()
    query = f"""
    SELECT
        book_id,
        title_clean AS title,
        description,
        ARRAY(
            SELECT SAFE_CAST(JSON_EXTRACT_SCALAR(author, '$.author_name') AS STRING)
            FROM UNNEST(authors_flat) AS author
        ) AS authors,
        average_rating,
        ratings_count
    FROM `{project}.{dataset}.goodreads_books_cleaned`
    WHERE book_id = {book_id}
    LIMIT 1
    """
    job = client.query(query)
    rows = [dict(row) for row in job.result()]   # FIXED
    return rows[0] if rows else None


# ---------------------------------------------------------------------
# GET BOOKS USER HAS READ
# ---------------------------------------------------------------------
def get_books_read_by_user(user_id: str):
    client, project = _get_client()
    query = f"""
    SELECT
        inter.book_id,
        b.title_clean AS title,
        b.average_rating,
        b.ratings_count
    FROM `{project}.{dataset}.goodreads_interactions_cleaned` AS inter
    LEFT JOIN `{project}.{dataset}.goodreads_books_cleaned` AS b
        ON inter.book_id = b.book_id
    WHERE inter.user_id_clean = '{user_id}'
      AND inter.is_read = TRUE
    """
    job = client.query(query)
    return [dict(row) for row in job.result()]   # FIXED


# ---------------------------------------------------------------------
# GET BOOKS USER HAS NOT READ
# ---------------------------------------------------------------------
def get_books_not_read_by_user(user_id: str):
    client, project = _get_client()
    query = f"""
    WITH read_books AS (
        SELECT book_id
        FROM `{project}.{dataset}.goodreads_interactions_cleaned`
        WHERE user_id_clean = '{user_id}'
          AND is_read = TRUE
    )
    SELECT
        b.book_id,
        b.title_clean AS title,
        b.average_rating,
        b.ratings_count
    FROM `{project}.{dataset}.goodreads_books_cleaned` AS b
    WHERE b.book_id NOT IN (SELECT book_id FROM read_books)
    ORDER BY b.average_rating DESC
    LIMIT 50
    """
    job = client.query(query)
    return [dict(row) for row in job.result()]   # FIXED
