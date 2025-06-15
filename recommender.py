import re
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def imdb_rating(df, id_column='movieId', rating_column='rating', q=0.5):

    mean_ratings = df.groupby(id_column).mean()[rating_column]
    mean_report = df[rating_column].mean()
    count = df.groupby(id_column).size()
    min_votes = count.quantile(q)
    if min_votes == 0:
        min_votes = 1
    movies_to_keep = count.where(count >= min_votes).dropna().index
    score = mean_ratings * count / (min_votes + count) + mean_report * min_votes / (min_votes + count)
    return score.loc[movies_to_keep].sort_index()


def pre_process_movie_file(path='movies_metadata.csv'):

    movie_file = pd.read_csv(path)

    # first, remove rows that don't have a valid id or imdb_id, no valid overview too
    movie_file = movie_file[~(movie_file['overview'].isna() & movie_file['tagline'].isna())]

    # now take care of the id columns
    movie_file.id = pd.to_numeric(movie_file['id'], errors='coerce')
    movie_file = movie_file.dropna(subset=['id', 'title'])
    movie_file.id = movie_file.id.astype(int)

    return movie_file.fillna(' ')


def collaborative_recommender(ratings_df, movie_df, top_k_users=10, top_n_movies=10, min_movies_rated=200, frac=0.25):
    # Count number of movies rated by each user
    rated_counts = ratings_df.groupby('userId').size()

    # Filter users with sufficient ratings
    filtered_users = rated_counts[rated_counts >= min_movies_rated].index
    ratings_redux = ratings_df[ratings_df.userId.isin(filtered_users)]

    # filter randomly over some movies in order to reduce the user-movies matrix due to memory constraints
    random_movies = pd.Series(ratings_redux.movieId.unique()).sample(frac=frac).values
    ratings_redux = ratings_redux[ratings_redux.movieId.isin(random_movies)]

    # Create user-item matrix
    user_movie_matrix = ratings_redux.pivot_table(index='userId', columns='movieId', values='rating').fillna(0.0)

    # Exit early if there aren't enough users
    if len(user_movie_matrix) < top_k_users + 1:
        print("Not enough users with sufficient ratings for recommendation.")
        return

    # Create sparse matrix for memory efficiency
    sparse_matrix = csr_matrix(user_movie_matrix.values)

    # Fit NearestNeighbors model
    knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=top_k_users + 1)
    knn.fit(sparse_matrix)

    # Select a random user
    random_idx = np.random.choice(user_movie_matrix.shape[0])
    random_user_id = user_movie_matrix.index[random_idx]
    print(f"Randomly selected user: {random_user_id}")

    # Find k nearest neighbors (including self)
    distances, indices = knn.kneighbors(sparse_matrix[random_idx], n_neighbors=top_k_users + 1)

    # Remove self from neighbors
    similar_user_indices = indices[0][1:]
    similar_user_ids = user_movie_matrix.index[similar_user_indices]

    # Movies already rated by the target user
    rated_by_user = user_movie_matrix.loc[random_user_id]
    already_rated = rated_by_user[rated_by_user > 0].index

    # Compute mean ratings from similar users
    similar_users_ratings = user_movie_matrix.loc[similar_user_ids]
    candidate_scores = similar_users_ratings.mean(axis=0)

    # Remove already rated movies
    candidate_scores = candidate_scores.drop(already_rated, errors='ignore')

    # Get top recommended movie IDs
    recommended_movie_ids = candidate_scores.sort_values(ascending=False).head(top_n_movies).index

    # Map to movie titles
    recommended_titles = movie_df[movie_df.id.isin(recommended_movie_ids)].title.values

    print(f"\nTop {top_n_movies} recommended movies for user {random_user_id}:")
    for title in recommended_titles:
        print(f"- {title}")


def content_based_recommender(movie_file, imdb_scores):

    # filter down to those movies that have had sufficient ratings for the imdb scoring system
    movie_df = movie_file[movie_file.id.isin(imdb_scores.index)]

    # now create a description column that will concatenate the overview and the taglines
    description = movie_df.loc[:, 'title'] + ' ' + movie_df.loc[:, 'overview'] + ' ' + movie_df.loc[:, 'tagline']
    description.index = movie_df.id

    # now, we use a simple system for tokenizing the description of the movies
    vectorizer = TfidfVectorizer(stop_words='english')

    # pre process the text
    x = vectorizer.fit_transform(description)

    # now, create the cosine similarity
    similarity = pd.DataFrame(cosine_similarity(x), index=movie_df.id, columns=movie_df.id)
    title_map = movie_df.set_index('id')['title']

    # now we select a random movie and get similar movies to that
    random_id = np.random.choice(similarity.index)
    random_title = title_map[random_id]
    print(f'The randomly chosen movie was {random_title}')

    # getting the similar movies
    similar_titles = similarity.loc[random_id].sort_values(ascending=False).iloc[1:11].index
    recommended_titles = title_map.loc[similar_titles]

    # printing the recommendations
    print(f'Recommended movies were {recommended_titles.values}')



def main():
    movie_file = pre_process_movie_file(path='movies_metadata.csv')
    ratings = pd.read_csv('ratings.csv')

    # calculate a weighted score for movies that is fairer based on imdb's rating system
    # this already filters movies for a low number of ratings
    imdb_score = imdb_rating(ratings, 'movieId', 'rating')

    # now, do a simple content based filtering that will produce recommendations of similar movies to a random one
    content_based_recommender(movie_file, imdb_score)

    # now, do a simple collaborative based filtering recommending movies to a random user based on similar users
    collaborative_recommender(ratings, movie_file)


if __name__ == '__main__':
    main()
