import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load data
columns = ['user_id', 'item_id', 'rating', 'timestamp']
ratings = pd.read_csv('u.data', sep='\t', names=columns)

movie_cols = ['item_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
              'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
              'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
              'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

movies = pd.read_csv('u.item', sep='|', names=movie_cols, encoding='latin-1')[['item_id', 'title']]

# Merge datasets
data = pd.merge(ratings, movies, on='item_id')

# Create user-item matrix
user_movie_matrix = data.pivot_table(index='user_id', columns='title', values='rating')

# Fill NaN with 0
user_movie_matrix = user_movie_matrix.fillna(0)

# Compute similarity between movies
movie_similarity = cosine_similarity(user_movie_matrix.T)
movie_similarity_df = pd.DataFrame(movie_similarity, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)

# Save model and data
with open('movie_similarity_model.pkl', 'wb') as f:
    pickle.dump(movie_similarity_df, f)

user_movie_matrix.to_pickle('user_movie_matrix.pkl')

print("✅ Model training complete. Files saved: 'movie_similarity_model.pkl' and 'user_movie_matrix.pkl'")
