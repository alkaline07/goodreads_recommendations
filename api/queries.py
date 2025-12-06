from google.cloud import bigquery
from .database import get_bq_client
from .generate_predictions import GeneratePredictions


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
    """
    Return model-generated recommendations (MF or BT) using GeneratePredictions.
    No fallback to BQ predictions table.
    """

    generator = GeneratePredictions()
    df = generator.get_predictions(user_id)

    # If model returns NO recommendations, return empty list.
    # (This will only happen if model can't score this user.)
    if df is None or len(df) == 0:
        return []

    # Convert DataFrame → list of dicts for main.py
    results = []
    for _, row in df.iterrows():
        results.append({
            "book_id": row["book_id"],
            "title": row["title"],
            "author": row.get("author_names", "Unknown"),
            "predicted_rating": row["rating"]
        })

    return results


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
    -- First extract authors from books_cleaned
    WITH exploded_authors AS (
        SELECT
            b.book_id,
            CAST(JSON_EXTRACT_SCALAR(a, '$.author_id') AS INT64) AS author_id
        FROM `{project}.{dataset}.goodreads_books_cleaned` b,
        UNNEST(b.authors_flat) a
    ),
    enriched_authors AS (
        SELECT
            ea.book_id,
            ARRAY_AGG(auth.name IGNORE NULLS)[OFFSET(0)] AS author
        FROM exploded_authors ea
        LEFT JOIN `{project}.{dataset}.goodreads_book_authors` auth
            ON ea.author_id = auth.author_id
        GROUP BY ea.book_id
    )

    SELECT
        inter.book_id,
        b.title_clean AS title,
        b.average_rating,
        b.ratings_count,
        ea.author,
        CASE
            WHEN inter.read_at_clean = 'Unknown' THEN NULL
            ELSE FORMAT_TIMESTAMP('%A, %B %d %Y',
                PARSE_TIMESTAMP('%a %b %d %H:%M:%S %z %Y', inter.read_at_clean)
            ) 
        END AS date_read
    FROM `{project}.{dataset}.goodreads_interactions_cleaned` inter
    LEFT JOIN `{project}.{dataset}.goodreads_books_cleaned` b
        ON inter.book_id = b.book_id
    LEFT JOIN enriched_authors ea
        ON inter.book_id = ea.book_id
    WHERE inter.user_id_clean = '{user_id}'
      AND inter.is_read = TRUE
    ORDER BY inter.read_at_clean DESC
    """

    job = client.query(query)
    return [dict(row) for row in job.result()]



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
    ),
    exploded_authors AS (
        SELECT
            b.book_id,
            CAST(JSON_EXTRACT_SCALAR(a, '$.author_id') AS INT64) AS author_id
        FROM `{project}.{dataset}.goodreads_books_cleaned` b,
        UNNEST(b.authors_flat) a
    ),
    enriched_authors AS (
        SELECT
            ea.book_id,
            ARRAY_AGG(auth.name IGNORE NULLS)[OFFSET(0)] AS author
        FROM exploded_authors ea
        LEFT JOIN `{project}.{dataset}.goodreads_book_authors` auth
            ON ea.author_id = auth.author_id
        GROUP BY ea.book_id
    )
    SELECT
        b.book_id,
        b.title_clean AS title,
        b.average_rating,
        b.ratings_count,
        ea.author
    FROM `{project}.{dataset}.goodreads_books_cleaned` b
    LEFT JOIN enriched_authors ea ON b.book_id = ea.book_id
    WHERE b.book_id NOT IN (SELECT book_id FROM read_books)
    ORDER BY b.average_rating DESC
    LIMIT 50
    """
    job = client.query(query)
    return [dict(row) for row in job.result()]



# ---------------------------------------------------------------------
# ENSURE GLOBAL TOP 10 TABLE EXISTS
# ---------------------------------------------------------------------
def ensure_global_top10_table_exists():
    client, project = _get_client()

    query = f"""
    BEGIN
      CREATE TABLE `{project}.{dataset}.global_top10_books` AS
      SELECT
          b.book_id,
          b.average_rating,
          b.title_clean,
          COALESCE(
              SAFE_CAST(JSON_EXTRACT_SCALAR(b.authors_flat[OFFSET(0)], '$.author_name') AS STRING),
              "Unknown"
          ) AS author,
          b.ratings_count
      FROM `{project}.{dataset}.goodreads_books_cleaned` AS b
      WHERE b.average_rating IS NOT NULL
      ORDER BY b.average_rating DESC, b.ratings_count DESC
      LIMIT 10;

    EXCEPTION WHEN ERROR THEN
      SELECT "Table already exists, skipping creation." AS status;

    END;
    """

    client.query(query).result()


# Run ONCE when module loads
ensure_global_top10_table_exists()
