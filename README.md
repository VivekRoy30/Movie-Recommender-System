# Movie Recommender System

A Streamlit web application that recommends movies based on a selected title
using a precomputed similarity model.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app expects `movies.pkl` and `similarity.pkl` to be present in the project
folder. Poster images are loaded from a configured TPDb-compatible URL first,
then TMDB, then Wikipedia/Wikimedia thumbnails as a no-key fallback.
Recommendations still render with placeholders if every poster service is
unavailable.

## Optional poster settings

```bash
export TMDB_API_KEY="your-tmdb-key"
export TPDB_API_KEY="your-tpdb-key"
export TPDB_POSTER_URL_TEMPLATE="https://example.com/poster?api_key={api_key}&tmdb_id={movie_id}&title={title}"
```

`TPDB_POSTER_URL_TEMPLATE` can return either an image directly or JSON with one
of these fields: `poster`, `poster_url`, `url`, `image`, or `image_url`.
