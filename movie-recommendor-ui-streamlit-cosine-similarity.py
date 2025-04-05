import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import ast
import re

# Page configuration
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

# Load movie dataset (with caching to improve performance)
@st.cache_data
def load_movie_data():
    movie_vectors = pd.read_csv("movie_vectors.csv")
    movie_vectors['title_lower'] = movie_vectors['title_y'].str.lower()
    return movie_vectors

# Function to parse string representation of numpy arrays
def parse_vector_string(vector_str):
    # Clean the string and convert to a proper list
    vector_str = re.sub(r'\s+', ' ', vector_str.replace('\n', ' ')).strip()
    # Use ast.literal_eval to convert the string to a list
    try:
        # Handle nested arrays (numpy array representation)
        if vector_str.startswith("np.str_"):
            # Extract the content within the quotes
            vector_content = re.search(r"'(.*)'", vector_str)
            if vector_content:
                vector_str = vector_content.group(1)
        
        # Convert string to list
        vector_list = [float(x) for x in vector_str.strip('[]').split()]
        return np.array(vector_list)
    except Exception as e:
        st.error(f"Error parsing vector: {e}")
        return None

def recommend_movies(movie_vectors, movie_title):
    # Convert title to lowercase
    movie_title = movie_title.lower()

    # Check if the movie exists
    movie_row = movie_vectors[movie_vectors['title_lower'] == movie_title]
    
    # Handle case where movie is not found
    if movie_row.empty:
        st.error(f"Movie '{movie_title}' not found in the database.")
        return None

    # Get the index of the target movie
    target_index = movie_row.index[0]
    
    # Parse vectors from string representation
    vectors = []
    for idx, row in movie_vectors.iterrows():
        vector = parse_vector_string(row['vector'])
        if vector is not None:
            vectors.append(vector)
        else:
            # Use a zero vector as fallback
            vectors.append(np.zeros(100))  # Adjust size if needed
    
    vectors = np.array(vectors)
    target_vector = vectors[target_index].reshape(1, -1)
    
    # Compute cosine similarity with all other movies
    similarities = cosine_similarity(target_vector, vectors)[0]

    # Get indices of top 10 most similar movies (excluding itself)
    similar_indices = similarities.argsort()[::-1]  # Sort in descending order
    similar_indices = [i for i in similar_indices if i != target_index]  # Remove the movie itself
    top_10_indices = similar_indices[:10]

    # Return the corresponding rows from the dataframe
    recommended_movies = movie_vectors.iloc[top_10_indices]
    
    return recommended_movies

def main():
    # Load the data
    movie_vectors = load_movie_data()

    # Create a centered column for the title and description
    col1, col2, col3 = st.columns([1,2,1])
    
    with col2:
        st.title("🎬 Movie Recommender")
        st.write("Find similar movies using cosine similarity!")

    # Movie title input
    movie_title = st.text_input("Enter a movie title:")

    # Recommendations will be triggered by entering a movie title
    if movie_title:
        # Get recommendations
        recommendations = recommend_movies(movie_vectors, movie_title)

        if recommendations is not None:
            st.header(f"Similar movies to '{movie_title}'")
                
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