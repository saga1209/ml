# Import libraries

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans


# Import necessary libraries (if not already imported)
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

df = pd.read_csv('netflix_titles.csv')
# Replace NaN with an empty string
df['description'] = df['description'].fillna('')


# Create a TfidfVectorizer and Remove stopwords
tfidf = TfidfVectorizer(stop_words='english')
# Fit and transform the data to a tfidf matrix
tfidf_matrix = tfidf.fit_transform(df['description'])
# Print the shape of the tfidf_matrix
tfidf_matrix.shape


# Compute the cosine similarity between each movie description
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


indices = pd.Series(df.index, index=df['title']).drop_duplicates()


def get_recommendations(title, cosine_sim=cosine_sim, num_recommend = 10):
    idx = indices[title]
# Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
# Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
# Get the scores of the 10 most similar movies
    top_similar = sim_scores[1:num_recommend+1]
# Get the movie indices
    movie_indices = [i[0] for i in top_similar]
# Return the top 10 most similar movies
    return df['title'].iloc[movie_indices]


get_recommendations('Power Rangers Zeo', num_recommend = 20)


def knn_recommendations(title, num_recommend=10):
    idx = indices[title]
    
    # Fit a KNN model to find the nearest neighbors
    knn = NearestNeighbors(n_neighbors=num_recommend + 1, metric='cosine')
    knn.fit(tfidf_matrix)
    
    distances, indices = knn.kneighbors(tfidf_matrix[idx], n_neighbors=num_recommend + 1)
    
    # Get the indices of the most similar movies
    movie_indices = indices.flatten()[1:]
    
    # Return the top 10 most similar movies
    return df['title'].iloc[movie_indices]

def kmeans_recommendations(title, num_recommend=10):
    idx = indices[title]
    
    # Fit a K-Means model to cluster similar movie descriptions
    num_clusters = 10  # You can adjust the number of clusters as needed
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(tfidf_matrix)
    
    cluster_label = kmeans.labels_[idx]
    similar_movies_indices = [i for i, label in enumerate(kmeans.labels_) if label == cluster_label]
    
    # Remove the input movie from the recommendations
    similar_movies_indices.remove(idx)
    
    # Return the top 10 most similar movies
    return df['title'].iloc[similar_movies_indices[:num_recommend]]


print("TF-IDF and Cosine Similarity Recommendations:")
print(get_recommendations('Power Rangers Zeo', num_recommend=20))

print("KNN Recommendations:")
print(knn_recommendations('Power Rangers Zeo', num_recommend=20))

print("K-Means Recommendations:")
print(kmeans_recommendations('Power Rangers Zeo', num_recommend=20))






# Load the dataset
df = pd.read_csv('netflix_titles.csv')
df['description'] = df['description'].fillna('')

# Create a TfidfVectorizer and remove stopwords
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['description'])

# Fit a K-Means model to cluster similar movie descriptions
num_clusters = 5  # You can adjust the number of clusters as needed
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
kmeans.fit(tfidf_matrix)

# Add the cluster labels to the DataFrame
df['cluster'] = kmeans.labels_

# Function to get recommendations from the same cluster
def kmeans_recommendations(title, num_recommend=10):
    idx = indices[title]
    cluster_label = df['cluster'].iloc[idx]

    # Get movies from the same cluster, excluding the input movie
    similar_movies = df[df['cluster'] == cluster_label]
    similar_movies = similar_movies[similar_movies.index != idx]

    # Sort by cluster similarity
    similar_movies['similarity'] = kmeans.transform(tfidf_matrix)[idx]
    similar_movies = similar_movies.sort_values(by='similarity')

    # Return the top 10 most similar movies
    return similar_movies.head(num_recommend)

print("K-Means Recommendations:")
print(kmeans_recommendations('Power Rangers Zeo', num_recommend=10))


