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

def accumulate_rating_counts(ratings):
    movie_interactions = defaultdict(int)
    max_user_id = 0
    for r in ratings:
        movie_interactions[r.movie_id] += 1
        max_user_id = max(max_user_id, r.user_id)

    return max_user_id, movie_interactions

def generate_dataset(max_user_id, movie_interactions):
    for m, count in movie_interactions.iteritems():
        interacting_users = random.sample(xrange(max_user_id + 1), count)
        for u in interacting_users:
            yield (u + 1, m + 1, 1.0, 1)

def parseargs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-ratings-fl",
                        type=str,
                        required=True,
                        help="Input ratings file")

    parser.add_argument("--output-ratings-fl",
                        type=str,
                        required=True,
                        help="Output ratings file")

    return parser.parse_args()


if __name__ == "__main__":
    args = parseargs()

    input_ratings = read_ratings(args.input_ratings_fl)
    max_user_id, interaction_counts = accumulate_rating_counts(input_ratings)

    with open(args.output_ratings_fl, "w") as output_fl:
        for t in generate_dataset(max_user_id, interaction_counts):
            line_contents = "::".join(map(str, t))
            output_fl.write(line_contents)
            output_fl.write("\n")


    
