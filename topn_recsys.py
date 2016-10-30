"""
Copyright 2016 Ronald J. Nowling

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
from sklearn.metrics import roc_auc_score

import numpy as np
import scipy.sparse as sp

from common import read_ratings

def create_training_sets(ratings, n_training, n_testing):
    print "Creating user movie-interaction lists"
    
    user_interactions = defaultdict(set)
    max_movie_id = 0
    for r in ratings:
        user_interactions[r.user_id].add(r.movie_id)
        max_movie_id = max(max_movie_id, r.movie_id)


    user_interactions = list(user_interactions.values())
    sampled_indices = random.sample(xrange(len(user_interactions)), n_training + n_testing)
    
    users = []
    movies = []
    interactions = []
    for new_user_id, idx in enumerate(sampled_indices[:n_training]):
        users.extend([new_user_id] * len(user_interactions[idx]))
        movies.extend(user_interactions[idx])
        interactions.extend([1.] * len(user_interactions[idx]))

    n_movies = max_movie_id + 1
    training_matrix = sp.coo_matrix((interactions, (users, movies)),
                               shape=(n_training, n_movies)).tocsr()

    users = []
    movies = []
    interactions = []
    for new_user_id, idx in enumerate(sampled_indices[n_training:]):
        users.extend([new_user_id] * len(user_interactions[idx]))
        movies.extend(user_interactions[idx])
        interactions.extend([1.] * len(user_interactions[idx]))

    n_movies = max_movie_id + 1
    testing_matrix = sp.coo_matrix((interactions, (users, movies)),
                               shape=(n_testing, n_movies)).tocsr()

    print training_matrix.shape, testing_matrix.shape

    return training_matrix, testing_matrix

def train_and_score(training, testing):
    print "Training and scoring"
    n_users = training.shape[0]
    predicted_scores = training.sum(axis=0) / float(n_users)
    print predicted_scores.shape

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
        
        for idx in indices:
            all_predicted_scores.append(predicted_scores[0, idx])
        all_labels.extend(labels)
            
    print len(all_labels), len(all_predicted_scores)

    auc = roc_auc_score(all_labels, all_predicted_scores)

    print "AUC", auc

def parseargs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ratings-fl",
                        type=str,
                        required=True,
                        help="Ratings file")

    parser.add_argument("--training",
                        type=int,
                        default=10000,
                        help="Number of training samples")

    parser.add_argument("--testing",
                        type=int,
                        default=1000,
                        help="Number of testing samples")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parseargs()

    ratings = read_ratings(args.ratings_fl)

    training, testing = create_training_sets(ratings, args.training, args.testing)

    train_and_score(training,
                    testing)


    
