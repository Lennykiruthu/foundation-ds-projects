import streamlit as st
import pickle
import requests
import pandas as pd
import numpy as np

with open('data_and_similarities.pkl', 'rb') as f:
    movies, similarities = pickle.load(f)

def get_recommendations(target_title, similarities, df, top_n=10):
  # Check if the target title exists in the DataFrame
  matching_movies = df[df['title_y'].str.lower() == target_title.lower()]
  if matching_movies.empty:
    print(f"{target_title} not found in dataset.")
    return []

  idx = matching_movies.index[0]
  sim_scores = list(enumerate(similarities[idx]))
  sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
  sim_scores = sim_scores[1:top_n+1]
  movie_indices = [i[0] for i in sim_scores]
  return df['title_y'].iloc[movie_indices]

def fetch_poster(movie_id):
    api_key = '75dd717f5963ac20dad1d3fe7680dc01'
    url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}'
    response = requests.get(url)
    data = response.json()
    poster_path = data['poster_path']
    full_path = f'https://image.tmdb.org/t/p/w500{poster_path}'
    return full_path

st.title('Movie Recommendation System')

selected_movie = st.selectbox('Select a movie:', movies['title_y'].values)

def find_movie_id_by_title(title, movies_df):
    # Find the row in movies_df where title matches and get the id
    matching_movie = movies_df[movies_df['title_y'] == title]
    if not matching_movie.empty:
        return matching_movie.iloc[0]['id']
    else:
        return None  # Or a default ID if no match found

if selected_movie:
    recommendations = get_recommendations(selected_movie, similarities, movies)
    st.write('Top 10 recommended movies:')
    
    # Display up to 10 recommendations in a grid (5 per row)
    for i in range(0, min(10, len(recommendations)), 5):
        cols = st.columns(5)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(recommendations):
                # Get the movie title directly from the Series
                movie_title = recommendations.iloc[idx]
                # Get the movie id for poster fetching
                movie_id = find_movie_id_by_title(movie_title, movies)  
                
                poster_url = fetch_poster(movie_id)
                with col:
                    st.image(poster_url, width=130)
                    st.write(movie_title)