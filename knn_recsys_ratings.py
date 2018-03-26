"""
Copyright 2018 Ronald J. Nowling

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
from collections import defaultdict
import random

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import Imputer

import numpy as np
import scipy.sparse as sp

def read_ratings(flname):
    user_ids = []
    movie_ids = []
    ratings = []
    with open(flname) as fl:
        # skip header
        next(fl)
        
        for ln in fl:
            user_id, movie_id, rating, timestamp = ln.strip().split(",")
            user_ids.append(int(user_id))
            movie_ids.append(int(movie_id))
            ratings.append(float(rating))

    n_movies = max(movie_ids) + 1
    n_users = max(user_ids) + 1

    ratings_matrix = np.zeros((n_users,
                               n_movies))

    for user_id, movie_id, rating in zip(user_ids, movie_ids, ratings):
        ratings_matrix[user_id, movie_id] = rating

    return ratings_matrix


def train_and_score(metric, training, testing, ks):
    print "Training and scoring"
    scores = []
    knn = NearestNeighbors(metric=metric, algorithm="brute")
    knn.fit(training)
    for k in ks:
        print "Evaluating for", k, "neighbors"
        neighbor_indices = knn.kneighbors(testing,
                                          n_neighbors=k,
                                          return_distance=False)

        all_predicted_scores = []
        all_labels = []
        for user_id in xrange(testing.shape[0]):
            user_row = testing[user_id, :]
            
            _, interaction_indices = user_row.nonzero()
            interacted = set(interaction_indices)
            non_interacted = set(xrange(testing.shape[1])) - interacted

            n_samples = min(len(non_interacted), len(interacted))
            sampled_interacted = random.sample(interacted, n_samples)
            sampled_non_interacted = random.sample(non_interacted, n_samples)

            indices = list(sampled_interacted)
            indices.extend(sampled_non_interacted)
            labels = [1] * n_samples
            labels.extend([0] * n_samples)
            
            neighbors = training[neighbor_indices[user_id, :], :]
            predicted_scores = neighbors.mean(axis=0)
            for idx in indices:
                all_predicted_scores.append(predicted_scores[0, idx])
            all_labels.extend(labels)

        print len(all_labels), len(all_predicted_scores)

        auc = roc_auc_score(all_labels, all_predicted_scores)

        print "k", k, "AUC", auc

def parseargs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ratings-fl",
                        type=str,
                        required=True,
                        help="Ratings file")
    
    parser.add_argument("--metric",
                        type=str,
                        choices=["euclidean", "cosine"],
                        default="euclidean",
                        help="Distance metric")

    parser.add_argument("--k",
                        type=int,
                        required=True,
                        help="Number of neigbhors")
    

    return parser.parse_args()


if __name__ == "__main__":
    args = parseargs()

    # read ratings
    print "Reading ratings"
    ratings_matrix = read_ratings(args.ratings_fl)

    n_users = ratings_matrix.shape[0]
    n_movies = ratings_matrix.shape[1]
    n_training_users = int(0.8 * n_users)
    
    # split test / train
    print "Splitting test / train"
    train_ids = random.sample(xrange(n_users),
                              n_training_users)
    test_ids = set(xrange(n_users)) - set(train_ids)
    test_ids = list(test_ids)
    n_test_users = len(test_ids)
    
    training_matrix = ratings_matrix[train_ids, :]
    testing_matrix = ratings_matrix[test_ids, :]

    # impute unknown ratings
    print "Imputing values"
    imputer = Imputer(missing_values=0)
    training_imputed_matrix = imputer.fit_transform(training_matrix)
    testing_imputed_matrix = imputer.transform(testing_matrix)

    # imputing culls columns with zero values so we need
    # to chop down the original matrices
    selected_columns = []
    for movie_id in xrange(n_movies):
        if not np.isnan(imputer.statistics_[movie_id]):
            selected_columns.append(movie_id)
            
    training_matrix = training_matrix[:, selected_columns]
    testing_matrix = testing_matrix[:, selected_columns]

    n_remaining_movies = training_matrix.shape[1]

    # perform predictions
    print "Performing kNN search"
    knn = NearestNeighbors(metric=args.metric)
    knn.fit(training_imputed_matrix)

    # returns n_test_users x k matrix
    neighbor_indices = knn.kneighbors(testing_imputed_matrix,
                                      n_neighbors=args.k,
                                      return_distance=False)

    # compute average ratings for each user
    print "Computing average ratings"
    predicted_ratings = np.zeros((n_test_users,
                                  n_remaining_movies))

    for user_id in xrange(n_test_users):
        neighbors = neighbor_indices[user_id, :]
        predicted_ratings[user_id, :] = np.average(training_imputed_matrix[neighbors, :], axis=0)

    # compute RMSE only for movies that have been rated
    print "Computing RMSE"
    squared_error = 0.0
    n = 0
    for user_id in xrange(n_test_users):
        nonzero_ratings = []
        for movie_id in xrange(n_remaining_movies):
            if testing_matrix[user_id, movie_id] > 0.0:
                nonzero_ratings.append(movie_id)

        squared_error += np.sum((testing_matrix[user_id, nonzero_ratings] - predicted_ratings[user_id, nonzero_ratings]) ** 2)
        n += len(nonzero_ratings)

    rmse = np.sqrt(squared_error / n)

    print "Root Mean-Squared Error:", rmse
    
    

    
