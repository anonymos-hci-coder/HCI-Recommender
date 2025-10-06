import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from surprise import Dataset, Reader, accuracy, SVD
from surprise.model_selection import train_test_split
import sys
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
import os


def build_model():

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'diverse_users.csv')
    
    test_users = pd.read_csv(csv_path)
    test_user_ids = test_users['userId'].unique().tolist()
    
    print('Building model...')

    # Load movies data from ./ml-32m/
    data_folder = os.path.join(script_dir, 'ml-32m')

    movies = pd.read_csv(f'{data_folder}/movies.csv')
    tags = pd.read_csv(f'{data_folder}/tags.csv')
    ratings = pd.read_csv(f'{data_folder}/ratings.csv')
    # limit ratings to user ratings that have rated more that 55 movies
    # it also filters the number of movies we can keep-- the reason is my
    # laptop limited power.
    ratings_f = ratings.groupby('userId').filter(lambda x: len(x) >= 50)
    ratings_f = ratings_f[~ratings_f['userId'].isin(test_user_ids)] # remove test users

    movie_list_rating = ratings_f.movieId.unique().tolist()
    # filter the movies data frame
    movies = movies[movies.movieId.isin(movie_list_rating)]
    # map movie to id:
    Mapping_file = dict(zip(movies.title.tolist(), movies.movieId.tolist()))
    # remove unnecessary timesteps
    tags.drop(['timestamp'], axis=1, inplace=True)
    ratings_f.drop(['timestamp'], axis=1, inplace=True)
    # make a useful dataframe from tags and movies
    mixed = pd.merge(movies, tags, on='movieId', how='left')

    # create metadata from all tags and genres
    mixed.fillna("", inplace=True)
    mixed = pd.DataFrame(mixed.groupby('movieId')['tag'].apply(
                                lambda x: "%s" % ' '.join(x)))
    Final = pd.merge(movies, mixed, on='movieId', how='left')
    Final['metadata'] = Final[['tag', 'genres']].apply(
                                lambda x: ' '.join(x), axis=1)

    # text transformation and truncated SVD to create a content latent matrix:
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(Final['metadata'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=Final.index.tolist())
    svd = TruncatedSVD(n_components=200)
    latent_matrix_1 = svd.fit_transform(tfidf_df)
    latent_matrix_1_df = pd.DataFrame(
                             latent_matrix_1,
                             index=Final.title.tolist())
    
    print('Creating latent matrix!')

    # text transformation and truncated SVD to create a collaborative
    # latent matrix:
    ratings_f1 = pd.merge(movies['movieId'], ratings_f,
                          on="movieId", how="right")
    
    user_enc = LabelEncoder()
    movie_enc = LabelEncoder()

    user_indices = user_enc.fit_transform(ratings_f1['userId'])
    movie_indices = movie_enc.fit_transform(ratings_f1['movieId'])
    data = ratings_f1['rating'].values

    print('Sparse matrix...')
    ratings_f2 = csr_matrix((data, (movie_indices, user_indices)))

    svd = TruncatedSVD(n_components=200)
    latent_matrix_2 = svd.fit_transform(ratings_f2)
    latent_matrix_2_df = pd.DataFrame(
                             latent_matrix_2,
                             index=Final.title.tolist())

    # now a user collabortive model using Surprise
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_f1[['userId', 'movieId', 'rating']],
                                reader)
    
    trainset = data.build_full_trainset()
    algorithm = SVD()
    # Train the algorithm on the trainset, and predict ratings for the testset
    algorithm.fit(trainset)
    # accuracy.rmse(algorithm.test(testset))

    print('Saving to files...')

    model_files = os.path.join(script_dir, 'Files')

    if not os.path.isdir(model_files):
        os.mkdir(model_files)

    # pickle all necessary files in ./Files/:
    ratings_f.to_pickle(f'{model_files}/rating.pkl')
    latent_matrix_1_df.to_pickle(f'{model_files}/latent_content.pkl')
    latent_matrix_2_df.to_pickle(f'{model_files}/latent_collaborative.pkl')
    with open(f'{model_files}/map.pkl', 'wb') as f:
        pickle.dump(Mapping_file, f, pickle.HIGHEST_PROTOCOL)
    with open(f'{model_files}/model_svd.pkl', 'wb') as f:
        pickle.dump(algorithm, f, pickle.HIGHEST_PROTOCOL)

    print('DONE!!!')

if __name__ == "__main__":
    build_model()
