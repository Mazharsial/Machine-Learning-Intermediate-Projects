import streamlit as st
import pickle
import pandas as pd

# Load model and data
movie_similarity = pickle.load(open('movie_similarity_model.pkl', 'rb'))
user_movie_matrix = pd.read_pickle('user_movie_matrix.pkl')

st.title("🎬 Movie Recommendation System")
st.write("Get movie recommendations based on your favorite film!")

# Movie selection
movie_list = user_movie_matrix.columns
selected_movie = st.selectbox("Select a movie:", movie_list)

def recommend(movie):
    if movie not in movie_similarity.columns:
        return []
    similar_scores = movie_similarity[movie].sort_values(ascending=False)[1:6]
    return list(similar_scores.index)

if st.button("Show Recommendations"):
    recommendations = recommend(selected_movie)
    st.subheader("Top 5 Recommended Movies:")
    for rec in recommendations:
        st.write(f"🎥 {rec}")
