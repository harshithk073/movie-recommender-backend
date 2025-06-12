from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load your movie data
movies_data = pd.read_csv('movies.csv')

# Prepare the data
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

combined_features = (
    movies_data['genres'] + ' ' +
    movies_data['keywords'] + ' ' +
    movies_data['tagline'] + ' ' +
    movies_data['cast'] + ' ' +
    movies_data['director']
)

vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
similarity = cosine_similarity(feature_vectors)

def parse_genres(genres_str):
    # Try to parse genres if it's a list-like string (e.g. "[{'id': 28, 'name': 'Action'}, ...]")
    try:
        genres = ast.literal_eval(genres_str)
        if isinstance(genres, list):
            return ", ".join([g['name'] for g in genres if isinstance(g, dict) and 'name' in g])
    except Exception:
        pass
    return genres_str

def get_recommendations(movie_name, num_recommendations=5):
    list_of_all_titles = movies_data['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
    if not find_close_match:
        return []
    close_match = find_close_match[0]
    index_of_the_movie = movies_data[movies_data.title == close_match].index[0]
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
    recommended_movies = []
    for i in sorted_similar_movies:
        movie = movies_data.iloc[i[0]]
        year = ""
        if pd.notnull(movie.release_date) and str(movie.release_date).strip():
            year = str(movie.release_date)[:4]
        genres = parse_genres(movie.genres)
        recommended_movies.append({
            "title": movie.title,
            "tagline": movie.tagline if pd.notnull(movie.tagline) else "",
            "genres": genres,
            "year": year,
            "rating": movie.vote_average if pd.notnull(movie.vote_average) else "",
            "director": movie.director if pd.notnull(movie.director) else "",
            "overview": movie.overview if pd.notnull(movie.overview) else ""
        })
    return recommended_movies

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    movie_title = data.get("movie_title", "")
    if not movie_title:
        return jsonify({"error": "No movie_title provided"}), 400
    recommendations = get_recommendations(movie_title)
    return jsonify({"recommendations": recommendations})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
