import streamlit as st
import pandas as pd

# Page configuration
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

# Load movie dataset (with caching to improve performance)
@st.cache_data
def load_movie_data():
    movies_df = pd.read_csv("movies_with_clusters.csv")
    movies_df['title_lower'] = movies_df['title_y'].str.lower()
    return movies_df

def recommend_movies(movies_df, movie_title):
    # Convert title to lowercase
    movie_title = movie_title.lower()

    # Check if the movie exists
    movie_row = movies_df[movies_df['title_lower'] == movie_title]
    
    # Handle case where movie is not found
    if movie_row.empty:
        st.error(f"Movie '{movie_title}' not found in the database.")
        return None

    # Get the cluster of the given movie
    movie_cluster = movie_row['DBSCAN_cluster_id'].values[0]

    # Find movies in the same cluster (excluding the input movie)
    similar_movies = movies_df[
        (movies_df["DBSCAN_cluster_id"] == movie_cluster) & 
        (movies_df['title_lower'] != movie_title)
    ]

    # If no similar movies found, return a message
    if similar_movies.empty:
        st.warning("No similar movies found in the same cluster.")
        return None

    # Select first 10 movies in the cluster
    recommended_movies = similar_movies.head(10)
    
    return recommended_movies

def main():
    # Load the data
    movies_df = load_movie_data()

    # Create a centered column for the title and description
    col1, col2, col3 = st.columns([1,2,1])
    
    with col2:
        st.title("🎬 Movie Recommender")
        st.write("Find similar movies in the same cluster!")

    # Movie title input
    movie_title = st.text_input("Enter a movie title:")

    # Recommendations will be triggered by entering a movie title
    if movie_title:
        # Get recommendations
        recommendations = recommend_movies(movies_df, movie_title)

        if recommendations is not None:
            st.header(f"Movies in the Same Cluster as '{movie_title}'")
                
            # Display recommendations in a grid
            for i in range(0, len(recommendations), 2):
                cols = st.columns(2)
                    
                # First movie in the column
                with cols[0]:
                    movie1 = recommendations.iloc[i]
                    st.subheader(movie1['title_y'])
                    st.divider()
                    
                # Second movie in the column (if exists)
                if i + 1 < len(recommendations):
                    with cols[1]:
                        movie2 = recommendations.iloc[i + 1]
                        st.subheader(movie2['title_y'])
                        st.divider()

# Run the app
if __name__ == "__main__":
    main()