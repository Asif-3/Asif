import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

@st.cache_resource
def load_data():
    """Load and preprocess the dataset."""
    try:
        df = pd.read_csv("dataset.csv")
        
        # Check if required columns exist in the dataset
        required_columns = ['Title', 'Overview', 'Poster_Url', 'Vote_Average', 'Genre']
        if not all(col in df.columns for col in required_columns):
            st.error(f"❌ Missing required columns: {', '.join(set(required_columns) - set(df.columns))}")
            return pd.DataFrame()
        
        # Rename columns for consistency
        df.rename(columns={
            'Title': 'title',
            'Overview': 'overview',
            'Poster_Url': 'poster_url',
            'Vote_Average': 'rating',  # Rating column
            'Genre': 'genre'
        }, inplace=True)

        # Drop rows with missing 'title' or 'overview'
        df.dropna(subset=['title', 'overview'], inplace=True)

        # Ensure 'rating' is numeric, coerce invalid values to NaN
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        
        # Drop rows where 'rating' is NaN (invalid entries)
        df.dropna(subset=['rating'], inplace=True)
        
        # Reset index after cleaning
        return df.reset_index(drop=True)

    except FileNotFoundError:
        st.error("❌ dataset.csv not found.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        return pd.DataFrame()

@st.cache_resource
def compute_similarity(df):
    """Compute the similarity matrix based on movie overviews."""
    try:
        tfidf = TfidfVectorizer(stop_words='english')
        matrix = tfidf.fit_transform(df['overview'].fillna(''))  # Fill missing overviews with empty strings
        return cosine_similarity(matrix)
    except Exception as e:
        st.error(f"❌ Error computing similarity: {e}")
        return None

def recommend(movie_title, df, similarity_matrix):
    """Get movie recommendations based on cosine similarity."""
    try:
        # Get the index of the selected movie
        idx = df[df['title'] == movie_title].index[0]
        distances = list(enumerate(similarity_matrix[idx]))
        
        # Sort movies based on similarity and get top 5 recommendations (excluding the selected movie)
        sorted_distances = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]
        recommendations = [
            (df.iloc[i].title, df.iloc[i].overview, df.iloc[i].poster_url)
            for i, _ in sorted_distances
        ]
        return recommendations
    except IndexError:
        st.error("❌ Movie not found in the dataset.")
        return []
    except Exception as e:
        st.error(f"❌ Error generating recommendations: {e}")
        return []

if 'show' not in st.session_state:
    st.session_state['show'] = False

movies = load_data()

if not movies.empty and 'title' in movies.columns:
    # Extract genres for the filter
    all_genres = sorted({genre.strip() for genres in movies['genre'].dropna() for genre in genres.split(',')})
    
    st.title("🎬 Smart Movie Recommender")
    st.caption("Built with love by Asif")

    # Filters
    selected_genres = st.multiselect("🎭 Filter by Genre(s):", all_genres)
    min_rating = st.slider("⭐ Minimum Rating:", 0.0, 10.0, 6.0, step=0.1)

    # Apply filters to the dataset
    filtered_movies = movies.copy()

    if selected_genres:
        filtered_movies = filtered_movies[filtered_movies['genre'].apply(
            lambda x: any(genre.strip() in x for genre in selected_genres)
        )]
    
    # Ensure that the rating filter works only if the rating is valid
    filtered_movies = filtered_movies[filtered_movies['rating'] >= min_rating]

    if filtered_movies.empty:
        st.warning("⚠️ No movies match the selected filters.")
    else:
        similarity = compute_similarity(filtered_movies)

        if similarity is not None:
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
            st.error("❌ Similarity matrix could not be computed. Please check the data and try again.")
else:
    st.warning("⚠️ Dataset is empty or invalid.")