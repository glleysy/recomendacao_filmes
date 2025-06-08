import re
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


def imdb_rating(df, id_column='movieId', rating_column='rating', q=0.5):

    mean_ratings = df.groupby(id_column).mean()[rating_column]
    mean_report = df[rating_column].mean()
    count = df.groupby(id_column).size()
    min_votes = count.quantile(q)
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


def main():
    movie_file = pre_process_movie_file(path='movies_metadata.csv')
    ratings = pd.read_csv('ratings.csv')

    # calculate a weighted score for movies that is fairer based on imdb's rating system
    # this already filters movies for a low number of ratings
    imdb_score = imdb_rating(ratings, 'movieId', 'rating')

    # now, filter the movie file to the valid movies with their newly calculated scores
    movie_file_redux = movie_file[movie_file.id.isin(imdb_score.index)]

    # now create a description column that will concatenate the overview and the taglines
    description = movie_file_redux.loc[:, 'overview'] + ' ' + movie_file_redux.loc[:, 'tagline']
    description.index = movie_file_redux.id

    # now, we use a simple system for tokenizing the description of the movies
    vectorizer = TfidfVectorizer(stop_words='english')

    # pre process the text
    x = vectorizer.fit_transform(description)

    # now, create the cosine similarity
    similarity = pd.DataFrame(cosine_similarity(x), index=movie_file_redux.title, columns=movie_file_redux.title)

    # now we select a random movie and get similar movies to that
    random_movie = np.random.choice(movie_file_redux.title, 1)[0]
    print(f'The randomly chosen movie was {random_movie}')

    # now get the similarity measures for that movie
    similarity_movie = similarity.loc[random_movie].sort_values(ascending=False).iloc[1:11].index

    # now print the titles of the movies
    recommended_movies = movie_file_redux.where(movie_file_redux.title.isin(similarity_movie)).dropna().title

    print(f'Recommended movies were {recommended_movies.values}')


if __name__ == '__main__':
    main()