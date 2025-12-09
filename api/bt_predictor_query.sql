WITH
active_users AS (
  SELECT DISTINCT user_id_clean
  FROM `{project_id}.{dataset}.goodreads_features`
  WHERE user_id_clean = @user_id
),
 
available_books AS (
  SELECT DISTINCT book_id
  FROM `{project_id}.{dataset}.goodreads_features`
  WHERE book_id IS NOT NULL 
    -- Add limits if necessary to prevent sending 100k rows to the endpoint
),
 
user_features AS (
  SELECT DISTINCT
    user_id_clean,
    num_books_read,
    avg_rating_given,
    user_activity_count,
    recent_activity_days,
    user_avg_reading_time_days
  FROM `{project_id}.{dataset}.goodreads_features`
  WHERE user_id_clean = @user_id
),
 
book_features AS (
  SELECT DISTINCT
    book_id,
    title_clean,
    average_rating,
    adjusted_average_rating,
    great,
    ratings_count,
    log_ratings_count,
    popularity_score,
    book_popularity_normalized,
    num_genres,
    is_series,
    title_length_in_characters,
    title_length_in_words,
    description_length,
    num_pages,
    publication_year,
    book_age_years,
    book_length_category,
    book_era,
    avg_pages_per_day,
    avg_book_reading_time_days,
    num_readers_with_reading_time,
    reading_pace_category
  FROM `{project_id}.{dataset}.goodreads_features`
  WHERE book_id IS NOT NULL
),

book_author_names AS (
  SELECT 
    CAST(b.book_id AS STRING) AS book_id,
    STRING_AGG(a.name, ', ') AS author_names
  FROM `{project_id}.{dataset}.goodreads_books_mystery_thriller_crime` b
  CROSS JOIN UNNEST(b.authors) AS author_struct
  JOIN `{project_id}.{dataset}.goodreads_book_authors` a
    ON CAST(a.author_id AS STRING) = CAST(author_struct.author_id AS STRING)
  GROUP BY 1
),
 
read_books AS (
  SELECT DISTINCT
    user_id,
    book_id
  FROM `{project_id}.{dataset}.goodreads_interactions_mystery_thriller_crime`
  WHERE is_read = TRUE
    AND user_id = @user_id 
),
 
unread_pairs AS (
  SELECT
    u.user_id_clean,
    b.book_id
  FROM active_users u
  CROSS JOIN available_books b
  WHERE NOT EXISTS (
    SELECT 1
    FROM read_books r
    WHERE r.user_id = u.user_id_clean
      AND r.book_id = b.book_id
  )
),
 
unread_features AS (
  SELECT
    up.user_id_clean,
    up.book_id,
    COALESCE(uf.num_books_read, 0) AS num_books_read,
    COALESCE(uf.avg_rating_given, 3.0) AS avg_rating_given,
    COALESCE(uf.user_activity_count, 0) AS user_activity_count,
    COALESCE(uf.recent_activity_days, 365) AS recent_activity_days,
    COALESCE(uf.user_avg_reading_time_days, 14.0) AS user_avg_reading_time_days,
    COALESCE(bf.title_clean, '') AS title_clean,
    COALESCE(bf.average_rating, 0.0) AS average_rating,
    COALESCE(bf.adjusted_average_rating, 0.0) AS adjusted_average_rating,
    COALESCE(bf.great, FALSE) AS great,
    COALESCE(bf.ratings_count, 0) AS ratings_count,
    COALESCE(bf.log_ratings_count, 0.0) AS log_ratings_count,
    CAST(COALESCE(bf.popularity_score, 0.0) AS INT64) AS popularity_score,
    COALESCE(bf.book_popularity_normalized, 0.0) AS book_popularity_normalized,
    COALESCE(bf.num_genres, 0) AS num_genres,
    COALESCE(bf.is_series, FALSE) AS is_series,
    COALESCE(bf.title_length_in_characters, 0) AS title_length_in_characters,
    COALESCE(bf.title_length_in_words, 0) AS title_length_in_words,
    COALESCE(bf.description_length, 0) AS description_length,
    COALESCE(bf.num_pages, 300) AS num_pages,
    COALESCE(bf.publication_year, 2000) AS publication_year,
    COALESCE(bf.book_age_years, 20) AS book_age_years,
    COALESCE(bf.book_length_category, 'medium') AS book_length_category,
    COALESCE(bf.book_era, 'contemporary') AS book_era,
    COALESCE(bf.avg_pages_per_day, 25.0) AS avg_pages_per_day,
    COALESCE(bf.avg_book_reading_time_days, 14.0) AS avg_book_reading_time_days,
    COALESCE(bf.num_readers_with_reading_time, 0) AS num_readers_with_reading_time,
    COALESCE(bf.reading_pace_category, 'moderate') AS reading_pace_category,
    COALESCE(uf.avg_rating_given, 3.0) - COALESCE(bf.adjusted_average_rating, 0.0) AS user_avg_rating_vs_book,
    COALESCE(
      SAFE_DIVIDE(14.0, NULLIF(bf.avg_book_reading_time_days, 0)),
      1.0
    ) AS user_reading_speed_ratio,
    COALESCE(
      SAFE_DIVIDE(bf.num_pages, 14.0),
      25.0
    ) AS user_pages_per_day_this_book
  FROM unread_pairs up
  LEFT JOIN user_features uf ON up.user_id_clean = uf.user_id_clean
  LEFT JOIN book_features bf ON up.book_id = bf.book_id
  WHERE uf.user_id_clean IS NOT NULL
    AND bf.book_id IS NOT NULL
),
 
predictions AS (
  SELECT
    pred.user_id_clean,
    pred.book_id,
    pred.predicted_rating,
    feat.title_clean,
    feat.average_rating,
    feat.ratings_count,
    feat.num_pages,
    feat.publication_year,
    feat.book_popularity_normalized,
    feat.book_length_category,
    feat.book_era,
    ROW_NUMBER() OVER (
      PARTITION BY pred.user_id_clean
      ORDER BY pred.predicted_rating DESC
    ) AS rank
  FROM ML.PREDICT(
    MODEL `{model_name}`,
    (
      SELECT
        num_books_read,
        avg_rating_given,
        user_activity_count,
        recent_activity_days,
        user_avg_reading_time_days,
        title_clean,
        average_rating,
        adjusted_average_rating,
        great,
        ratings_count,
        log_ratings_count,
        popularity_score,
        book_popularity_normalized,
        num_genres,
        is_series,
        title_length_in_characters,
        title_length_in_words,
        description_length,
        num_pages,
        publication_year,
        book_age_years,
        book_length_category,
        book_era,
        avg_pages_per_day,
        avg_book_reading_time_days,
        num_readers_with_reading_time,
        reading_pace_category,
        user_avg_rating_vs_book,
        user_reading_speed_ratio,
        user_pages_per_day_this_book,
        user_id_clean,
        book_id
      FROM unread_features
    )
  ) AS pred
  INNER JOIN unread_features feat
    ON pred.user_id_clean = feat.user_id_clean
    AND pred.book_id = feat.book_id
),
 
distinct_predictions AS (
  SELECT
    p.user_id_clean,
    p.book_id,
    p.title_clean,
    auth.author_names,
    p.predicted_rating,
    p.rank,
    p.average_rating,
    p.ratings_count,
    p.num_pages,
    p.publication_year,
    p.book_popularity_normalized,
    p.book_length_category,
    p.book_era,
    ROW_NUMBER() OVER (
      PARTITION BY p.title_clean, auth.author_names
      ORDER BY p.predicted_rating DESC
    ) AS title_author_rank,
    ROW_NUMBER() OVER (
      ORDER BY p.predicted_rating DESC
    ) AS distinct_rank
  FROM predictions p
  LEFT JOIN book_author_names auth
    ON CAST(p.book_id AS STRING) = auth.book_id
  WHERE TRUE
)

SELECT
  user_id_clean,
  book_id,
  COALESCE(author_names, 'Unknown Author') AS author_names,
  predicted_rating,
  distinct_rank as rank,
  title_clean as title,
  average_rating as rating,
  ratings_count,
  num_pages,
  publication_year,
  book_popularity_normalized,
  book_length_category,
  book_era
FROM distinct_predictions
WHERE title_author_rank = 1
  AND distinct_rank <= 50
ORDER BY distinct_rank;