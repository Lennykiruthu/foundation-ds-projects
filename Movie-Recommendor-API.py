import uvicorn
import pandas as pd
from fastapi import FastAPI, HTTPException

# Initialize FastAPI app
app = FastAPI()

# Load movie dataset
movies_df = pd.read_csv("movies_with_clusters.csv")  # Ensure this file has movie titles and cluster IDs

# Convert movie titles to lowercase for case-insensitive matching
movies_df['title_lower'] = movies_df['title_y'].str.lower()

@app.get("/")
def home():
    return {"message": "Welcome to the Movie Recommendation API! Use /recommend/{movie_title} to get recommendations."}

@app.get("/recommend/{movie_title}")
def recommend_movies(movie_title: str):
    # Convert title to lowercase
    movie_title = movie_title.lower()

    # Check if the movie exists
    movie_row = movies_df[movies_df['title_lower'] == movie_title]
    if movie_row.empty:
        raise HTTPException(status_code=404, detail=f"Movie '{movie_title}' not found in the dataset")

    # Get the cluster of the given movie by converting the single row dataframe into a numpy array and accessing the first and only value.
    movie_cluster = movie_row['DBSCAN_cluster_id'].values[0]

    # Find movies in the same cluster (excluding the input movie)
    similar_movies = movies_df[
        (movies_df["DBSCAN_cluster_id"] == movie_cluster) & 
        (movies_df['title_lower'] != movie_title)
    ]['title_y'].tolist()

    # If no similar movies found, return a helpful message
    if not similar_movies:
        return {"input_movie": movie_title, "recommended_movies": [], "message": "No similar movies found in the same cluster."}

    # Return up to 5 similar movies
    return {"input_movie": movie_title, "recommended_movies": similar_movies[:5]}

# Run the app
if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)