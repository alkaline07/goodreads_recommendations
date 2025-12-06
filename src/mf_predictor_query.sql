WITH 
  -- 0. Get Books Already Read by User
  read_books AS (
    SELECT DISTINCT
      user_id_clean,
      CAST(book_id AS STRING) AS book_id
    FROM `{project_id}.{dataset}.goodreads_features`
    WHERE is_read = TRUE
        AND user_id_clean = @user_id
  ),
  -- 1. Generate Raw Recommendations from ML Model
  raw_recommendations AS (
    SELECT
      user_id_clean,
      CAST(book_id AS STRING) AS book_id,
      predicted_rating
    FROM ML.RECOMMEND(
        MODEL `{model_name}`,
        (SELECT @user_id AS user_id_clean)
    )
  ),
  -- 2. Filter out books already read by the user
  filtered_recommendations AS (
    SELECT r.user_id_clean, r.book_id, r.predicted_rating
    FROM `raw_recommendations` AS r
    WHERE
      NOT EXISTS(
        SELECT 1
        FROM `read_books` AS rb
        WHERE rb.user_id_clean = r.user_id_clean AND rb.book_id = r.book_id
      )
  ),
  -- 3. Fetch Titles and Aggregate Authors for recommended books
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
        SELECT book_id FROM `filtered_recommendations`
    )
    GROUP BY b.book_id
  ),
  -- 4. Fetch Average Rating from Features Table for recommended books
  book_stats AS (
    SELECT DISTINCT
        CAST(book_id AS STRING) AS book_id_join_key,
        average_rating
    FROM `{project_id}.{dataset}.goodreads_features`
    WHERE CAST(book_id AS STRING) IN (
        SELECT book_id FROM `filtered_recommendations`
    )
  ),
  -- 5. Combine Filtered Recommendations, Metadata, and Stats
  combined AS (
    SELECT
      r.user_id_clean,
      r.book_id AS book_id,
      d.title AS title,
      COALESCE(d.author_names, 'Unknown Author') AS author_names,
      s.average_rating AS rating,
      r.predicted_rating,
      ROW_NUMBER() OVER (
          PARTITION BY d.title, COALESCE(d.author_names, 'Unknown Author')
          ORDER BY r.predicted_rating DESC
        ) AS title_author_rank,
      ROW_NUMBER() OVER (
        ORDER BY r.predicted_rating DESC
    ) AS distinct_rank
    FROM `filtered_recommendations` AS r
    LEFT JOIN `book_details` AS d
      ON r.book_id = d.book_id_join_key
    LEFT JOIN `book_stats` AS s
      ON r.book_id = s.book_id_join_key
  )
SELECT
    user_id_clean,
    book_id,
    title,
    author_names,
    rating,
    predicted_rating
FROM `combined`
WHERE title_author_rank = 1
  AND distinct_rank <= 50
ORDER BY distinct_rank;