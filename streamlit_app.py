import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

# Load data
data = pd.read_csv("books.csv", on_bad_lines='skip')
data['genre'] = data['genre'].fillna('Unknown')
data['rating'] = data['rating'].fillna(data['rating'].mean())
data['author'] = data['author'].fillna('Unknown')
data['title'] = data['title'].fillna('Unknown')
data['img'] = data['img'].fillna('Unknown')
books = data.drop('pages', axis=1)
books['title'] = books['title'].str.lower().str.strip()
books['author'] = books['author'].str.lower().str.strip()

# Preprocess combined features for content-based filtering
books['combined_features'] = books['title'] + " " + books['genre']
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books['combined_features'])
tfidf_matrix = csr_matrix(tfidf_matrix)

# Dimensionality reduction
if tfidf_matrix.shape[0] > 1:  # Ensure there are at least 2 samples
    #n_components = min(tfidf_matrix.shape) - 1
    n_components = 100
    svd = TruncatedSVD(n_components=n_components)
    tfidf_reduced = svd.fit_transform(tfidf_matrix)
else:
    st.error("Not enough data for dimensionality reduction.")
    tfidf_reduced = None  # Handle this case in further code

# Define content recommendation function
def get_content_recommendations(title, books, tfidf_reduced, top_n=10):
    matched_books = books[books['title'].str.contains(title, case=False, na=False)]
    if not matched_books.empty:
        book_idx = matched_books.index[0]
        cosine_sim = cosine_similarity([tfidf_reduced[book_idx]], tfidf_reduced).flatten()
        similar_indices = cosine_sim.argsort()[-top_n-1:-1][::-1]
        similar_books = books.iloc[similar_indices]
        return similar_books[['title', 'author', 'genre', 'rating']]
    return None

# Fuzzy match for similar titles
def get_similar_titles(title, books, limit=3):
    titles = books['title'].tolist()
    closest_matches = process.extract(title, titles, limit=limit)
    return [match[0] for match in closest_matches]

# Collaborative filtering data preparation
def prepare_collab_data(books):
    books['user_id'] = 0  # Dummy user for simplicity
    user_item_matrix = pd.pivot_table(books, values='rating', index='user_id', columns='title').fillna(0)
    sparse_matrix = csr_matrix(user_item_matrix.values)
    return sparse_matrix, user_item_matrix

# Train ALS model
def train_als_model(sparse_matrix):
    model = AlternatingLeastSquares(factors=50, regularization=0.1, iterations=10)
    model.fit(sparse_matrix.T)
    return model

# ALS recommendation
def get_als_recommendations(model, user_item_matrix, title, n_recommendations=5):
    if title not in user_item_matrix.columns:
        return None
    title_idx = user_item_matrix.columns.get_loc(title)
    scores = model.recommend(0, user_item_matrix.values[title_idx], N=n_recommendations)
    recommended_indices = [i[0] for i in scores]
    recommendations = user_item_matrix.columns[recommended_indices]
    return recommendations


# Hybrid recommendation
def get_hybrid_recommendations(title, genre, books, tfidf_reduced, als_model, user_item_matrix, n_recommendations=5):
    content_recommendations = get_content_recommendations(title, books, tfidf_reduced, top_n=n_recommendations)
    
    if content_recommendations is None:
        similar_titles = get_similar_titles(title, books)
        if similar_titles:
            title = similar_titles[0]
            content_recommendations = get_content_recommendations(title, books, tfidf_reduced, top_n=n_recommendations)

    collaborative_recommendations = get_als_recommendations(als_model, user_item_matrix, title, n_recommendations=n_recommendations)
    
    # Apply the fix here
    if collaborative_recommendations is None or len(collaborative_recommendations) == 0:
        collaborative_recommendations = []

    # Ensure content_recommendations and collaborative_recommendations are in correct format
    combined_recommendations = pd.concat(
        [content_recommendations, pd.DataFrame({'title': collaborative_recommendations})],
        ignore_index=True
    ).drop_duplicates()
    
    return combined_recommendations.head(n_recommendations) if not combined_recommendations.empty else "No recommendations available."


# Streamlit UI
st.title("Book Recommendation System")

# Get user input
title_input = st.text_input("Enter the book title:", key="title_input_key").lower().strip()
genre_input = st.text_input("Enter the book genre:", key="genre_input_key").lower().strip()

if st.button("Recommend"):
    # Prepare collaborative filtering data
    sparse_matrix, user_item_matrix = prepare_collab_data(books)
    als_model = train_als_model(sparse_matrix)

    # Check if user inputs are provided
    if not title_input:
        st.write("Please enter a book title.")
    elif not genre_input:  # Check if genre input is provided
        st.write("Please enter a book genre.")
    else:
        # Get hybrid recommendations
        recommendations = get_hybrid_recommendations(title_input, genre_input, books, tfidf_reduced, als_model, user_item_matrix)

        # Display recommendations
        if isinstance(recommendations, str):
            st.write(recommendations)
        else:
            st.write("Top Recommendations:")
            st.dataframe(recommendations)

# Option to show rating distribution
if st.checkbox("Show Rating Distribution"):
    plt.figure(figsize=(10, 6))
    sns.histplot(books['rating'], bins=20, kde=True)
    st.pyplot()
    plt.clf()  # Clear the figure to avoid overlap

# Option to show top authors
if st.checkbox("Show Top Authors"):
    plt.figure(figsize=(10, 6))
    top_authors = books['author'].value_counts().head(10)
    sns.barplot(x=top_authors.index, y=top_authors.values, palette='viridis')
    st.pyplot()
    plt.clf()  # Clear the figure to avoid overlap

# Option to show top genres
if st.checkbox("Show Top Genres"):
    plt.figure(figsize=(10, 6))
    top_genres = books['genre'].value_counts().head(10)
    sns.barplot(x=top_genres.index, y=top_genres.values, palette='plasma')
    st.pyplot()
    plt.clf()  # Clear the figure to avoid overlap
