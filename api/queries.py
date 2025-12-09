from google.cloud import bigquery
from .database import get_bq_client
from .generate_predictions import GeneratePredictions
from .log_click_event import LogClickEvent
from datetime import datetime
from datapipeline.scripts.logger_setup import get_logger

logger = get_logger("api-queries")

_client = None
_project = None
dataset = "books"


def _get_client():
    global _client, _project
    if _client is None:
        _client = get_bq_client()
        _project = _client.project
        logger.info("BigQuery client initialized", project=_project)
    return _client, _project

# ---------------------------------------------------------------------
# CHECK IF USER EXISTS IN PREDICTION TABLE
# ---------------------------------------------------------------------
def check_user_exists(user_id: str) -> bool:
    client, project = _get_client()
    query = f"""
    SELECT COUNT(*) AS cnt
    FROM `{project}.{dataset}.goodreads_features`
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
    logger.info("Fetching recommendations", user_id=user_id)
    generator = GeneratePredictions()
    df = generator.get_predictions(user_id)

    # If model returns NO recommendations, return empty list.
    # (This will only happen if model can't score this user.)
    if df is None or len(df) == 0:
        logger.warning("No recommendations found", user_id=user_id)
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
    logger.info("Recommendations fetched", user_id=user_id, count=len(results))
    return results


# ---------------------------------------------------------------------
# GET GLOBAL TOP 10 BOOKS FOR NEW USERS
# ---------------------------------------------------------------------
def get_global_top_recommendations():
    logger.info("Fetching global top recommendations")
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
def log_ctr_event(user_id: str, book_id: int, event_type: str = "click", book_title: str = None):
    """
    Wrapper around LogClickEvent so API does NOT change.
    """
    event_logger = LogClickEvent()
    try:
        result = event_logger.log_user_event(
            user_id=user_id,
            book_id=int(book_id),     # BigQuery table uses STRING
            event_type=event_type,
            book_title=book_title
        )
        logger.info("CTR event logged", user_id=user_id, book_id=book_id, event_type=event_type)
        return bool(result)

    except Exception as e:
        logger.error("CTR logging failed", error=str(e), user_id=user_id, book_id=book_id)
        return False


# ---------------------------------------------------------------------
# GET BOOKS USER HAS READ
# ---------------------------------------------------------------------
def get_books_read_by_user(user_id: str):
    logger.info("Fetching read books", user_id=user_id)
    client, project = _get_client()
    query = f"""
    -- First extract authors from books table
    WITH exploded_authors AS (
        SELECT
            b.book_id,
            a.author_id
        FROM `{project}.{dataset}.goodreads_books_mystery_thriller_crime` b,
        UNNEST(b.authors) a
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
        b.title as title ,
        b.average_rating,
        b.ratings_count,
        ea.author,
    FORMAT_TIMESTAMP(
        '%A, %B %d %Y',
        COALESCE(
            SAFE.PARSE_TIMESTAMP('%a %b %d %H:%M:%S %z %Y', NULLIF(inter.read_at, "")),
            TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 2 DAY)
        )
    ) AS date_read
    FROM `{project}.{dataset}.goodreads_interactions_mystery_thriller_crime` inter
    LEFT JOIN `{project}.{dataset}.goodreads_books_mystery_thriller_crime` b
        ON inter.book_id = b.book_id
    LEFT JOIN enriched_authors ea
        ON inter.book_id = ea.book_id
    WHERE inter.user_id = '{user_id}'
      AND inter.is_read = TRUE
    ORDER BY inter.read_at DESC
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
        FROM `{project}.{dataset}.goodreads_interactions_mystery_thriller_crime`
        WHERE user_id = '{user_id}'
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
# INSERT NEW BOOK INTO INTERACTIONS TABLE WHEN MARKED AS READ
# ---------------------------------------------------------------------
def insert_read_interaction(user_id: str, book_id: int, rating: int = None):
    client, project = _get_client()
    check_query = f"""
        SELECT book_id 
        FROM `{project}.{dataset}.goodreads_books_mystery_thriller_crime`
        WHERE book_id = {book_id}
        LIMIT 1
    """
    check_job = client.query(check_query)
    
    # If total_rows is 0, the book doesn't exist. Stop here.
    if check_job.result().total_rows == 0:
        logger.error("Books ID not found in database", book_id=book_id)
        return False
    
    table = f"{project}.{dataset}.goodreads_interactions_mystery_thriller_crime"

    now = datetime.utcnow().strftime("%a %b %d %H:%M:%S +0000 %Y")
    rating = rating if rating is not None else 0

    merge_query = f"""
    MERGE {table} T
    USING (
      SELECT 
        '{user_id}' as user_id, 
        {book_id} as book_id, 
        {rating} as rating, 
        '{now}' as now_ts
    ) S
    ON T.user_id = S.user_id AND T.book_id = S.book_id
    
    -- If the record exists, UPDATE it
    WHEN MATCHED THEN
      UPDATE SET 
        rating = S.rating,
        date_updated = S.now_ts,
        read_at = S.now_ts,
        date_added = S.now_ts,
        is_read = TRUE
        
    -- If the record does NOT exist, INSERT it
    WHEN NOT MATCHED THEN
      INSERT (user_id, book_id, review_id, date_added, date_updated, read_at, started_at, review_text_incomplete, rating, is_read)
      VALUES (S.user_id, S.book_id, NULL, S.now_ts, S.now_ts, S.now_ts, '', '', S.rating, TRUE)
    """

    try:
        query_job = client.query(merge_query)
        query_job.result()  # Wait for the job to complete
        print(f"Successfully upserted record for user {user_id} and book {book_id}")
        return True
    except Exception as e:
        print("BQ exception:", e)
        return False


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
