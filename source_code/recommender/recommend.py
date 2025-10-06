import pickle
import pandas as pd
import numpy as np
from source_code.recommender.model import ContentBased, CollabBased, HybridBased, ModelBased
from typing import Iterable, Optional, Dict
import os

class Recommender:
    def __init__(self, model_files):

        # Load mappings (title -> movieId)
        with open(f'{model_files}/map.pkl', 'rb') as f:
            mapping: Dict[str, int] = pickle.load(f)

        # Expose movie_map (backward compatible)
        self.movie_map: Dict[str, int] = mapping # title -> movieId
        self.title_to_id: Dict[str, int] = mapping # alias
        self.id_to_title: Dict[int, str] = {v: k for k, v in mapping.items()} # movieId -> title

        with open(f'{model_files}/model_svd.pkl', 'rb') as f:
            self.algo = pickle.load(f)
        with open(f'{model_files}/map.pkl', 'rb') as f:
            self.movie_map = pickle.load(f)
        with open(f'{model_files}/rating.pkl', 'rb') as f:
            self.rating = pickle.load(f)
        with open(f'{model_files}/latent_collaborative.pkl', 'rb') as f:
            latent_collab = pickle.load(f)
        with open(f'{model_files}/latent_content.pkl', 'rb') as f:
            latent_content = pickle.load(f)

        self.clf_content = ContentBased(latent_content)
        self.clf_collab = CollabBased(latent_collab)
        self.clf_hybrid = HybridBased(latent_content, latent_collab)
        self.clf_algo = ModelBased(self.algo)

    def get_all_recommendations(self, moviename, n):
        if moviename in self.movie_map.keys():
            output = {
                'content': {'content':
                            self.clf_content.predict_top_n(moviename, n)},
                'collaborative': {'collaborative':
                                  self.clf_collab.predict_top_n(moviename, n)},
                'hybrid': {'hybrid':
                           self.clf_hybrid.predict_top_n(moviename, n)},
                     }
        else:
            output = None
        return output
    
    def get_user_recommendation(self, user_id, n):
        if user_id in self.rating.userId.unique():
            ui_list = self.rating[
                self.rating.userId == user_id].movieId.tolist()
            d = {k: v for k, v in self.movie_map.items() if v not in ui_list}
            output = self.clf_algo.predict_top_n_user(user_id, d, n)
        else:
            output = None

        return output

    def predict_top_n_new_user(self, user_ratings: dict, d: dict, n=10):
        """
        For a NEW USER:
        user_ratings: dict {movieId: rating}
        d: dict of all candidate movies {title: movieId}
        """
        # SVD parameters
        pu = []  # latent user vector
        qi = self.algo.qi   # latent matrix for items
        movie_index = self.algo.trainset._raw2inner_id_items

        # movies from the dataset
        rated_movies = {m: r for m, r in user_ratings.items() if m in movie_index}

        if len(rated_movies) == 0:
            return []  # all movies are invalid

        # approximate the latent user using least squares
        X = []
        y = []
        for m, r in rated_movies.items():
            inner_id = movie_index[m]
            X.append(qi[inner_id])
            y.append(r)
        X = np.array(X)
        y = np.array(y)

        # least squares solution to find pu (latent user)
        pu, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

        # prediction for other movies
        predictions = []
        for title, mid in d.items():
            if mid in rated_movies:
                continue
            if mid not in movie_index:
                continue
            inner_id = movie_index[mid]
            score = np.dot(pu, qi[inner_id]) + self.algo.bi[inner_id] + self.algo.trainset.global_mean
            predictions.append((title, score))

        pdf = pd.DataFrame(predictions, columns=['movies', 'ratings'])
        pdf.sort_values('ratings', ascending=False, inplace=True)
        pdf.set_index('movies', inplace=True)
        return pdf.head(n).index.tolist()



script_dir = os.path.dirname(os.path.abspath(__file__))
recommender = Recommender(os.path.join(script_dir, 'Files'))

def get_recommendation(history: dict , number_of_recommendations = 10):
    # title -> movieId
    title_to_id = recommender.title_to_id

    # movieId -> rating
    user_ratings_mapped = {
        title_to_id[title]: rating
        for title, rating in history.items()
        if title in title_to_id
    }

    # all movies but user rated ones
    d = {title: mid for title, mid in title_to_id.items()
         if mid not in user_ratings_mapped.keys()}

    # getting recommendations
    output = recommender.predict_top_n_new_user(user_ratings_mapped, d, number_of_recommendations)
    return output
    pass
# if __name__ == '__main__':

#     # user ratings
#     new_user_ratings = {
#         'Inception (2010)': 5.0,
#         'Dark Knight, The (2008)': 5.0,
#         'Matrix, The (1999)': 5.0,
#         'Interstellar (2014)': 4.5,
#         'Hangover, The (2009)': 1.0,
#         'Dumb and Dumber (Dumb & Dumber) (1994)': 1.5   # not in movie titles 
#     }

    
#     output = get_recommendation(new_user_ratings)
#     print(output)
#     for l in output:
#         print(l)


