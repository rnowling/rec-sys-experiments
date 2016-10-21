"""
Proof of concept for a recommendation system implemented using Logistic Regression and feature hashing.  You'll need to download one of the MovieLens datasets from http://grouplens.org/datasets/movielens/.

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
from datetime import date
import random

from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

import numpy as np
import scipy.sparse as sp

from common import read_ratings
from common import plot_aucs_nnzs

def create_training_sets(ratings):
    print "Creating user movie-interaction lists"
    user_movies = defaultdict(set)
    max_movie_id = 0
    for r in ratings:
        user_movies[r.user_id].add(r.movie_id)
        max_movie_id = max(max_movie_id, r.movie_id)


    training_set = []
    test_set = []
    for user_id, movie_ids in user_movies.iteritems():
        if random.random() >= 0.5:
            training_set.append((user_id, movie_ids))
        else:
            test_set.append((user_id, movie_ids))

    return max_movie_id, training_set, test_set

def generate_features(max_movie_id, seen_movies):
    all_movie_ids = set(range(0, max_movie_id + 1))
    seen_pairs = []
    unseen_pairs = []

    # positive examples
    for movie_id1 in seen_movies:
        movie_pairs = dict()
        for movie_id2 in seen_movies:
            # product with itself will always be 1 when the label is 1
            # and so will result in model overfitting
            if movie_id1 != movie_id2:
                movie_pairs["%s_%s" % (movie_id1, movie_id2)] = 1.
        seen_pairs.append(movie_pairs)

    # negative_examples
    unseen_movies = all_movie_ids - seen_movies
    for movie_id1 in random.sample(unseen_movies, len(seen_movies)):
        movie_pairs = dict()
        for movie_id2 in seen_movies:
            movie_pairs["%s_%s" % (movie_id1, movie_id2)] = 1.
        unseen_pairs.append(movie_pairs)

    labels = np.hstack([np.ones(len(seen_pairs)), np.zeros(len(unseen_pairs))])

    return labels, (seen_pairs, unseen_pairs)
    

def train_and_score(max_movie_id, training, testset, model_sizes):
    extractors = dict()
    models = dict()
    
    print "Creating models"
    for model_size in model_sizes:
        extractors[model_size] = FeatureHasher(n_features=2**model_size)
        models[model_size] = SGDClassifier(loss="log", penalty="L2")

    print "Training"
    for i, (user_id, seen_movies) in enumerate(training):
        print "Training on user", i, user_id
        labels, (seen_pairs, unseen_pairs) = generate_features(max_movie_id, seen_movies)
        for model_size, extractor in extractors.iteritems():
            seen_features = extractor.transform(seen_pairs)
            unseen_features = extractor.transform(unseen_pairs)
            features = sp.vstack([seen_features, unseen_features])
            model = models[model_size]
            model.partial_fit(features, labels, classes=[0, 1])
                
    print "Testing"
    all_labels = []
    all_predicted_labels = defaultdict(list)
    all_predicted_prob = defaultdict(list)
    for i, (user_id, seen_movies) in enumerate(testset):
        print "Testing on user", i, user_id
        labels, (seen_pairs, unseen_pairs) = generate_features(max_movie_id, seen_movies)
        all_labels.extend(labels)
        
        for model_size, extractor in extractors.iteritems():
            seen_features = extractor.transform(seen_pairs)
            unseen_features = extractor.transform(unseen_pairs)
            features = sp.vstack([seen_features, unseen_features])
            
            model = models[model_size]
            predicted_labels = model.predict(features)
            predicted_prob = model.predict_proba(features)
            all_predicted_labels[model_size].extend(predicted_labels)
            # Probabilities for positive class
            all_predicted_prob[model_size].extend(predicted_prob[:, 1])

    print "Scoring"
    aucs = []
    nnz_features = []
    for model_size, model in models.iteritems():
        pred_log_prob = all_predicted_prob[model_size]
        auc = roc_auc_score(all_labels, pred_log_prob)
        cm = confusion_matrix(all_labels, all_predicted_labels[model_size])
        print "Model size", model_size, "auc", auc
        print cm
        print
        aucs.append(auc)
        nnz_features.append(np.count_nonzero(model.coef_))

    return aucs, nnz_features
        

def parseargs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ratings-fl",
                        type=str,
                        required=True,
                        help="Ratings file")

    parser.add_argument("--figures-dir",
                        type=str,
                        required=True,
                        help="Directory for outputting figures")

    parser.add_argument("--model-bits",
                        type=int,
                        nargs="+",
                        help="Model sizes in terms of bits")

    return parser.parse_args()


if __name__ == "__main__":
    args = parseargs()

    ratings = read_ratings(args.ratings_fl)
    max_movie_id, training, test = create_training_sets(ratings)

    training = random.sample(training, 1000)
    test = random.sample(test, 100)

    aucs, nnzs = train_and_score(max_movie_id, training, test, args.model_bits)
    
    plot_aucs_nnzs(args.figures_dir + "/lr_hashing_auc_nnzs.png",
                   args.model_bits,
                   aucs,
                   nnzs)
