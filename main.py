import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

# Load movie data from a CSV file
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
        # Drop rows with missing title or overview
        df.dropna(subset=['title', 'overview'], inplace=True)
        return df.reset_index(drop=True)
    except FileNotFoundError:
        st.error("❌ dataset.csv not found.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        return pd.DataFrame()

# Compute similarity matrix based on movie overviews
@st.cache_resource
def compute_similarity(df):
    tfidf = TfidfVectorizer(stop_words='english')
    matrix = tfidf.fit_transform(df['overview'].fillna(''))  # Fill missing overviews with empty string
    return cosine_similarity(matrix)

# Recommend movies based on a selected movie's title
def recommend(movie_title, df, similarity_matrix):
    try:
        # Normalize title (remove spaces and case-insensitivity)
        normalized_title = movie_title.strip().lower()
        df['normalized_title'] = df['title'].str.strip().str.lower()

        # Find the index of the selected movie
        idx = df[df['normalized_title'] == normalized_title].index[0]
        distances = list(enumerate(similarity_matrix[idx]))
        sorted_distances = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]

        # Return recommended movies with title, overview, and poster URL
        return [
            (df.iloc[i].title, df.iloc[i].overview, df.iloc[i].poster_url)
            for i, _ in sorted_distances
        ]
    except IndexError:
        st.error("❌ Selected movie not found in the dataset.")
        return []

# Initialize session state to manage recommendation visibility
if 'show' not in st.session_state:
    st.session_state['show'] = False

# Load the movie data
movies = load_data()

if not movies.empty and 'title' in movies.columns:
    # Extract all available genres
    all_genres = sorted({genre.strip() for genres in movies['genre'].dropna() for genre in genres.split(',')})
    
    st.title("🎬 Smart Movie Recommender")
    st.caption("Built with love by Asif")

    # Genre filter
    selected_genres = st.multiselect("🎭 Filter by Genre(s):", all_genres)
    # Minimum rating filter
    min_rating = st.slider("⭐ Minimum Rating:", 0.0, 10.0, 6.0, step=0.1)

    # Apply selected filters to the dataset
    filtered_movies = movies.copy()

    # Convert 'rating' to numeric and filter out rows with invalid ratings
    filtered_movies['rating'] = pd.to_numeric(filtered_movies['rating'], errors='coerce')
    filtered_movies = filtered_movies.dropna(subset=['rating'])

    # Filter by selected genres
    if selected_genres:
        filtered_movies = filtered_movies[filtered_movies['genre'].apply(
            lambda x: any(genre.strip() in x for genre in selected_genres)
        )]

    # Filter by minimum rating
    filtered_movies = filtered_movies[filtered_movies['rating'] >= min_rating]

    # Check if any movies are left after filtering
    if filtered_movies.empty:
        st.warning("⚠️ No movies match the selected filters.")
    else:
        # Compute the similarity matrix for the filtered movies
        similarity = compute_similarity(filtered_movies)

        # Movie selection dropdown
        selected_movie = st.selectbox("🎞️ Choose a movie to get recommendations:", sorted(filtered_movies['title'].unique()))

        # Recommend button
        if st.button("✨ Recommend"):
            st.session_state['show'] = True

        # Display recommendations if button is pressed
        if st.session_state['show']:
            st.subheader("You may also like:")

            recommendations = recommend(selected_movie, filtered_movies, similarity)

            if recommendations:
                # Display the recommendations in columns
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