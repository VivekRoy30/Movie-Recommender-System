from concurrent.futures import ThreadPoolExecutor
from html import escape
import os
from pathlib import Path
import pickle
from urllib.parse import quote

import pandas as pd
import requests
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MOVIES_FILE = BASE_DIR / "movies.pkl"
SIMILARITY_FILE = BASE_DIR / "similarity.pkl"
TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"
TMDB_API_KEY = os.getenv("TMDB_API_KEY", TMDB_API_KEY)
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"
TPDB_API_KEY = os.getenv("TPDB_API_KEY")
TPDB_POSTER_URL_TEMPLATE = os.getenv("TPDB_POSTER_URL_TEMPLATE")
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary"
POSTER_REQUEST_HEADERS = {"User-Agent": "movie-recommender-system/1.0"}


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
        response = requests.get(
            url,
            params=params,
            headers=POSTER_REQUEST_HEADERS,
            timeout=3,
        )
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
        response = requests.get(
            url,
            params=params,
            headers=POSTER_REQUEST_HEADERS,
            timeout=3,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
    except (requests.RequestException, ValueError):
        return None

    if not results:
        return None

    return build_poster_url(results[0].get("poster_path"))


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_poster_from_tpdb(movie_id, movie_title):
    """Return a poster URL from a configured TPDb-compatible endpoint.

    TPDb deployments/API wrappers vary, so the app accepts a URL template instead
    of assuming one endpoint shape. The template can use {movie_id}, {title}, and
    {api_key} placeholders.
    """
    if not TPDB_POSTER_URL_TEMPLATE:
        return None

    url = TPDB_POSTER_URL_TEMPLATE.format(
        api_key=TPDB_API_KEY or "",
        movie_id=movie_id,
        title=quote(str(movie_title)),
    )

    try:
        response = requests.get(url, headers=POSTER_REQUEST_HEADERS, timeout=4)
        response.raise_for_status()
    except requests.RequestException:
        return None

    content_type = response.headers.get("content-type", "")
    if content_type.startswith("image/"):
        return response.url

    try:
        payload = response.json()
    except ValueError:
        return None

    if isinstance(payload, dict):
        for key in ("poster", "poster_url", "url", "image", "image_url"):
            poster_url = payload.get(key)
            if poster_url:
                return poster_url

        data = payload.get("data")
        if isinstance(data, dict):
            for key in ("poster", "poster_url", "url", "image", "image_url"):
                poster_url = data.get(key)
                if poster_url:
                    return poster_url

        results = payload.get("results")
        if isinstance(results, list) and results:
            first_result = results[0]
            if isinstance(first_result, dict):
                for key in ("poster", "poster_url", "url", "image", "image_url"):
                    poster_url = first_result.get(key)
                    if poster_url:
                        return poster_url

    return None


def get_wikipedia_search_titles(movie_title):
    params = {
        "action": "query",
        "list": "search",
        "srsearch": f"{movie_title} film",
        "format": "json",
        "srlimit": 3,
    }

    try:
        response = requests.get(
            WIKIPEDIA_API_URL,
            params=params,
            headers=POSTER_REQUEST_HEADERS,
            timeout=4,
        )
        response.raise_for_status()
        results = response.json().get("query", {}).get("search", [])
    except (requests.RequestException, ValueError):
        return []

    return [result["title"] for result in results if result.get("title")]


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_poster_from_wikipedia(movie_title):
    """Return a poster thumbnail from Wikipedia/Wikimedia without an API key."""
    search_titles = get_wikipedia_search_titles(movie_title)
    candidate_titles = [
        f"{movie_title} (film)",
        *search_titles,
        str(movie_title),
    ]

    seen_titles = set()
    for candidate_title in candidate_titles:
        if candidate_title in seen_titles:
            continue

        seen_titles.add(candidate_title)
        url = f"{WIKIPEDIA_SUMMARY_URL}/{quote(candidate_title, safe='')}"
        try:
            response = requests.get(
                url,
                headers=POSTER_REQUEST_HEADERS,
                timeout=4,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError):
            continue

        if payload.get("type") == "disambiguation":
            continue

        thumbnail = payload.get("thumbnail", {})
        poster_url = thumbnail.get("source")
        if poster_url:
            return poster_url

    return None


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_poster(movie_id, movie_title):
    """Return a poster URL using TPDb/configured, TMDB, then no-key fallbacks."""
    return (
        fetch_poster_from_tpdb(movie_id, movie_title)
        or fetch_poster_by_id(movie_id)
        or fetch_poster_by_title(movie_title)
        or fetch_poster_from_wikipedia(movie_title)
    )


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
                "Recommendations are ready, but poster images could not be loaded. "
                "Please check your internet connection or poster API settings."
            )

        columns = st.columns(len(recommended_movies))

        for column, movie in zip(columns, recommended_movies):
            with column:
                if movie["poster"]:
                    st.image(movie["poster"], width="stretch")
                else:
                    render_poster_placeholder(movie["title"])

                st.markdown(
                    f'<div class="movie-card-title">{escape(str(movie["title"]))}</div>',
                    unsafe_allow_html=True,
                )
                st.caption(f"Similarity score: {movie['score']:.3f}")
