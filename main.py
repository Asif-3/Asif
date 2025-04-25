import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

@st.cache_resource
def load_data():
    try:
        df = pd.read_csv("dataset.csv")
        df.rename(columns={
            'Title': 'title',
            'Overview': 'overview',
            'Poster_Url': 'poster_url',
            'Vote_Average': 'rating',
            'Genre': 'genre'
        }, inplace=True)
        df.dropna(subset=['title', 'overview'], inplace=True)
        return df.reset_index(drop=True)
    except FileNotFoundError:
        st.error("❌ dataset.csv not found.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        return pd.DataFrame()

@st.cache_resource
def compute_similarity(df):
    tfidf = TfidfVectorizer(stop_words='english')
    matrix = tfidf.fit_transform(df['overview'].fillna(''))
    return cosine_similarity(matrix)

def recommend(movie_title, df, similarity_matrix):
    try:
        # Normalize the movie title to handle spaces and case differences
        normalized_title = movie_title.strip().lower()
        df['normalized_title'] = df['title'].str.strip().str.lower()

        # Find the movie index by matching the normalized title
        idx = df[df['normalized_title'] == normalized_title].index[0]
        distances = list(enumerate(similarity_matrix[idx]))
        sorted_distances = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]
        return [
            (df.iloc[i].title, df.iloc[i].overview, df.iloc[i].poster_url)
            for i, _ in sorted_distances
        ]
    except IndexError:
        st.error("❌ Selected movie not found.")
        return []

if 'show' not in st.session_state:
    st.session_state['show'] = False

movies = load_data()

if not movies.empty and 'title' in movies.columns:
    all_genres = sorted({genre.strip() for genres in movies['genre'].dropna() for genre in genres.split(',')})
    
    st.title("🎬 Smart Movie Recommender")
    st.caption("Built with love by Asif")

    # Filters
    selected_genres = st.multiselect("🎭 Filter by Genre(s):", all_genres)
    min_rating = st.slider("⭐ Minimum Rating:", 0.0, 10.0, 6.0, step=0.1)

    # Apply filters
    filtered_movies = movies.copy()

    # Convert 'rating' column to numeric, coercing errors
    filtered_movies['rating'] = pd.to_numeric(filtered_movies['rating'], errors='coerce')

    # Drop rows with missing or invalid ratings
    filtered_movies = filtered_movies.dropna(subset=['rating'])

    if selected_genres:
        filtered_movies = filtered_movies[filtered_movies['genre'].apply(
            lambda x: any(genre.strip() in x for genre in selected_genres)
        )]
    
    filtered_movies = filtered_movies[filtered_movies['rating'] >= min_rating]

    if filtered_movies.empty:
        st.warning("⚠️ No movies match the selected filters.")
    else:
        similarity = compute_similarity(filtered_movies)
        selected_movie = st.selectbox("🎞️ Choose a movie to get recommendations:", sorted(filtered_movies['title'].unique()))

        if st.button("✨ Recommend"):
            st.session_state['show'] = True

        if st.session_state['show']:
            st.subheader("You may also like:")

            recommendations = recommend(selected_movie, filtered_movies, similarity)

            if recommendations:
                cols = st.columns(len(recommendations))
                for i, (title, overview, poster_url) in enumerate(recommendations):
                    with cols[i]:
                        st.image(
                            poster_url if pd.notna(poster_url) else "https://via.placeholder.com/200x300?text=No+Image",
                            width=150
                        )
                        st.markdown(f"**{title}**")
                        st.caption(overview[:150] + "...")
            else:
                st.warning("⚠️ No recommendations found.")
else:
    st.warning("⚠️ Dataset is empty or invalid.")