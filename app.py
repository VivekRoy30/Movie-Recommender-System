from concurrent.futures import ThreadPoolExecutor
from html import escape
import os
from pathlib import Path
import pickle

import pandas as pd
import requests
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MOVIES_FILE = BASE_DIR / "movies.pkl"
SIMILARITY_FILE = BASE_DIR / "similarity.pkl"
TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"
TMDB_API_KEY = os.getenv("TMDB_API_KEY", TMDB_API_KEY)
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"


st.set_page_config(page_title="Movie Recommender", layout="wide")


@st.cache_resource(show_spinner="Loading recommendation data...")
def load_recommendation_data():
    """Load and validate the serialized movie data used by the app."""
    missing_files = [
        str(path.name)
        for path in (MOVIES_FILE, SIMILARITY_FILE)
        if not path.exists()
    ]
    if missing_files:
        raise FileNotFoundError(
            "Missing required model file(s): " + ", ".join(missing_files)
        )

    with MOVIES_FILE.open("rb") as movies_handle:
        movies_data = pd.DataFrame(pickle.load(movies_handle))

    with SIMILARITY_FILE.open("rb") as similarity_handle:
        similarity_data = pickle.load(similarity_handle)

    required_columns = {"movie_id", "title"}
    missing_columns = required_columns.difference(movies_data.columns)
    if missing_columns:
        raise ValueError(
            "movies.pkl is missing column(s): " + ", ".join(sorted(missing_columns))
        )

    if len(movies_data) != similarity_data.shape[0]:
        raise ValueError(
            "Movie data and similarity matrix have different lengths: "
            f"{len(movies_data)} movies vs {similarity_data.shape[0]} similarity rows."
        )

    return movies_data, similarity_data


def build_poster_url(poster_path):
    if not poster_path:
        return None

    return f"{POSTER_BASE_URL}{poster_path}"


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_poster_by_id(movie_id):
    """Return a poster URL using the TMDB movie id."""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {"api_key": TMDB_API_KEY, "language": "en-US"}

    try:
        response = requests.get(url, params=params, timeout=3)
        response.raise_for_status()
        poster_path = response.json().get("poster_path")
    except (requests.RequestException, ValueError):
        return None

    return build_poster_url(poster_path)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_poster_by_title(movie_title):
    """Fallback poster lookup for records whose stored TMDB id has no poster."""
    url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": movie_title, "language": "en-US"}

    try:
        response = requests.get(url, params=params, timeout=3)
        response.raise_for_status()
        results = response.json().get("results", [])
    except (requests.RequestException, ValueError):
        return None

    if not results:
        return None

    return build_poster_url(results[0].get("poster_path"))


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_poster(movie_id, movie_title):
    """Return a poster URL from TMDB, with an id lookup and title fallback."""
    return fetch_poster_by_id(movie_id) or fetch_poster_by_title(movie_title)


def recommend(movie_title, movies, similarity, count=5):
    matches = movies.index[movies["title"] == movie_title].tolist()
    if not matches:
        return []

    movie_index = matches[0]
    similar_movies = sorted(
        enumerate(similarity[movie_index]),
        reverse=True,
        key=lambda item: item[1],
    )[1 : count + 1]

    movie_rows = [movies.iloc[index] for index, _score in similar_movies]
    poster_requests = [
        (int(movie["movie_id"]), str(movie["title"]))
        for movie in movie_rows
    ]

    with ThreadPoolExecutor(max_workers=min(count, len(poster_requests))) as executor:
        poster_urls = list(
            executor.map(lambda request: fetch_poster(*request), poster_requests)
        )

    recommendations = []
    for movie, poster_url, (_index, score) in zip(movie_rows, poster_urls, similar_movies):
        recommendations.append(
            {
                "title": movie["title"],
                "poster": poster_url,
                "score": float(score),
            }
        )

    return recommendations


def render_poster_placeholder(title):
    st.markdown(
        f"""
        <div class="poster-placeholder">
            <span>{escape(str(title))}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .app-title {
            font-size: 2.4rem;
            font-weight: 750;
            margin-bottom: 0.2rem;
        }

        .app-subtitle {
            color: #5c6575;
            font-size: 1rem;
            margin-bottom: 1.5rem;
        }

        .movie-card-title {
            font-weight: 700;
            min-height: 3rem;
            margin: 0.65rem 0 0.35rem;
        }

        .poster-placeholder {
            align-items: center;
            aspect-ratio: 2 / 3;
            background: linear-gradient(145deg, #1f2937, #3b4758);
            border-radius: 8px;
            color: #ffffff;
            display: flex;
            font-weight: 700;
            justify-content: center;
            min-height: 260px;
            padding: 1rem;
            text-align: center;
        }

        div.stButton > button {
            width: 100%;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-title">Movie Recommender System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Pick a movie and discover five similar titles '
    "from the trained similarity model.</div>",
    unsafe_allow_html=True,
)

try:
    movies, similarity = load_recommendation_data()
except (FileNotFoundError, ValueError, pickle.UnpicklingError) as error:
    st.error(str(error))
    st.stop()

movie_list = sorted(movies["title"].dropna().unique())
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list,
    index=movie_list.index("Avatar") if "Avatar" in movie_list else 0,
)

if st.button("Show Recommendations", type="primary"):
    with st.spinner("Finding similar movies..."):
        recommended_movies = recommend(selected_movie, movies, similarity)

    if not recommended_movies:
        st.warning("No recommendations were found for this movie.")
    else:
        st.subheader("Recommended movies")
        poster_count = sum(1 for movie in recommended_movies if movie["poster"])
        if poster_count == 0:
            st.info(
                "Recommendations are ready, but TMDB posters could not be loaded. "
                "Please check your internet connection or TMDB API key."
            )

        columns = st.columns(len(recommended_movies))

        for column, movie in zip(columns, recommended_movies):
            with column:
                if movie["poster"]:
                    st.image(movie["poster"], use_container_width=True)
                else:
                    render_poster_placeholder(movie["title"])

                st.markdown(
                    f'<div class="movie-card-title">{escape(str(movie["title"]))}</div>',
                    unsafe_allow_html=True,
                )
                st.caption(f"Similarity score: {movie['score']:.3f}")
