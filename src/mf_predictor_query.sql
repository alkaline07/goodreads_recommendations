WITH 
-- 1. Generate Recommendations
recommendations AS (
  SELECT 
    user_id_clean, 
    book_id, 
    predicted_rating
  FROM ML.RECOMMEND(
    MODEL `{model_name}`,
    (SELECT @user_id AS user_id_clean)
  )
),

-- 2. Fetch Titles and Aggregate Authors 
book_details AS (
  SELECT 
    CAST(b.book_id AS STRING) AS book_id_join_key,
    ANY_VALUE(b.title) AS title,
    STRING_AGG(a.name, ', ') AS author_names
  FROM `{project_id}.{dataset}.goodreads_books_mystery_thriller_crime` b
  CROSS JOIN UNNEST(b.authors) AS author_struct
  JOIN `{project_id}.{dataset}.goodreads_book_authors` a
    ON CAST(a.author_id AS STRING) = CAST(author_struct.author_id AS STRING)
  WHERE CAST(b.book_id AS STRING) IN (
      SELECT CAST(book_id AS STRING) FROM recommendations
  )
  GROUP BY b.book_id
),

-- 3. Fetch Average Rating from Features Table
book_stats AS (
  SELECT DISTINCT
    CAST(book_id AS STRING) AS book_id_join_key,
    average_rating
  FROM `{project_id}.{dataset}.goodreads_features`
  WHERE CAST(book_id AS STRING) IN (
      SELECT CAST(book_id AS STRING) FROM recommendations
  )
)

-- 4. Combine Recommendations, Metadata, and Stats
SELECT 
  r.user_id_clean,
  r.book_id as book_id,
  d.title as title,
  COALESCE(d.author_names, 'Unknown Author') AS author_names,
  s.average_rating as rating,
  r.predicted_rating
FROM recommendations r
LEFT JOIN book_details d
  ON CAST(r.book_id AS STRING) = d.book_id_join_key
LEFT JOIN book_stats s
  ON CAST(r.book_id AS STRING) = s.book_id_join_key
ORDER BY r.predicted_rating DESC
LIMIT 50;