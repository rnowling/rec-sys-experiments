from collections import defaultdict
from collections import namedtuple
import csv

import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
import numpy as np

class GrowingHistogram(object):
    def __init__(self, bin_width):
        self.bin_width = float(bin_width)
        self.bins = defaultdict(int)

    def accumulate(self, value):
        bin_idx = int(np.floor(value / self.bin_width))
        self.bins[bin_idx] += 1

    def histogram(self):
        offset_idx = 0
        n_bins = max(self.bins.iterkeys()) + 1
        min_idx = min(self.bins.iterkeys)
        
        if min_idx < 0:
            offset_idx += np.abs(min_idx)
            n_bins += np.abs(min_idx)

        counts = [0] * n_bins
        lowerbounds = [0.] * (n_bins + 1)
        
        for bin_idx, count in self.bins.iteritems():
            idx = offset_idx + bin_idx
            counts[idx] = count
            lowerbounds[idx] = bin_idx * self.bin_width

        lowerbounds[-1] = n_bins * self.bin_width

        return counts, lowerbounds

def read_ratings(flname):
    Rating = namedtuple("Rating", ["user_id", "movie_id", "rating", "timestamp"])
    with open(flname) as fl:
        for ln in fl:
            user_id, movie_id, rating, timestamp = ln.strip().split("::")
            yield Rating(user_id=int(user_id) - 1,
                         movie_id=int(movie_id) - 1,
                         rating=float(rating),
                         timestamp=int(timestamp))
            

def plot_correlation(flname, title, x_label, y_label, dataset):
    """
    Scatter plot with line of best fit

    dataset - tuple of (x_values, y_values)
    """
    plt.clf()
    plt.hold(True)
    plt.scatter(dataset[0], dataset[1], alpha=0.7, color="k")
    xs = np.array(dataset[0])
    ys = np.array(dataset[1])
    A = np.vstack([xs, np.ones(len(xs))]).T
    m, c = np.linalg.lstsq(A, ys)[0]
    plt.plot(xs, m*xs + c, "c-")
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    plt.xlim([0.25, max(dataset[0])])
    plt.ylim([10., max(dataset[1])])
    plt.title(title, fontsize=18)
    plt.savefig(flname, DPI=200)

def plot_histogram(flname, title, x_label, datasets):
    """
    Histogram

    dataset - list tuples of (frequencies, bins, style)
    """
    plt.clf()
    plt.hold(True)
    max_bin = 0
    for frequencies, bins, style in datasets:
        xs = []
        ys = []
        max_bin = max(max_bin, max(bins))
        for i, f in enumerate(frequencies):
            xs.append(bins[i])
            xs.append(bins[i+1])
            ys.append(f)
            ys.append(f)
        plt.plot(xs, ys, style)
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel("Occurrences", fontsize=16)
    plt.xlim([0, max_bin + 1])
    plt.title(title, fontsize=18)
    plt.savefig(flname, DPI=200)

def plot_aucs_nnzs(flname, model_bits, aucs, nzs):
    fig, ax1 = plt.subplots()
    ax1.plot(model_bits, aucs, 'c-')
    ax1.set_xlabel('Hashed Features (log_2)', fontsize=16)
    # Make the y-axis label and tick labels match the line color.
    ax1.set_ylabel('AUC', color='c', fontsize=16)
    for tl in ax1.get_yticklabels():
        tl.set_color('c')

    ax2 = ax1.twinx()
    ax2.plot(model_bits, nzs, 'k-')
    ax2.set_ylabel('Non-zero Weights', color='k', fontsize=16)
    for tl in ax2.get_yticklabels():
        tl.set_color('k')

    fig.subplots_adjust(right=0.8)
    fig.savefig(flname, DPI=200)
