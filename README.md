# Movie Recommender System

A Streamlit web application that recommends movies based on a selected title
using a precomputed similarity model.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app expects `movies.pkl` and `similarity.pkl` to be present in the project
folder. Poster images are loaded from TMDB when available; recommendations still
render with placeholders if the poster service is unavailable.
