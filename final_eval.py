import pickle
import numpy as np
from scipy.spatial import distance

def get_histogram(predictions, bins):
    
    sample_size = predictions.shape[0]
    dataset_size = predictions.shape[1]
    
    histograms = []
    for x_idx in range(dataset_size):
        samples = []
        for sample_idx in range(sample_size):
            samples.append(predictions[sample_idx][x_idx][0])
        hist, intervals = np.histogram(samples, bins)

        histograms.append(hist)

    return histograms

def get_jensen_shannon_dist(H1, H2):
    n = len(H1)
    dists = []
    for i in range(n):
        d = distance.jensenshannon(H1[i], H2[i])
        dists.append(d)

    return np.array(dists)

# Read the prediction results
bootstrap = pickle.load(open("bootstrap_predictions", "rb"))
all_dropout = pickle.load(open("all_dropout_predictions", "rb"))
last_dropout = pickle.load(open("last_dropout_predictions", "rb"))

bins = [i*0.01 for i in range(101)]
print("Getting all histograms")
H1_bootstrap = get_histogram(bootstrap, bins)
H2_all_dropout = get_histogram(all_dropout, bins)
H3_last_dropout = get_histogram(last_dropout, bins)
print("Completed histograms")

print("JSD calculation- bootstrap vs all_dropout")
JSD_H1_H2 = get_jensen_shannon_dist(\
        H1_bootstrap,
        H2_all_dropout)

print(np.average(JSD_H1_H2))
print(np.max(JSD_H1_H2))
print(np.min(JSD_H1_H2))

print("JSD calculation- bootstrap vs last_dropout")
JSD_H1_H3 = get_jensen_shannon_dist(\
        H1_bootstrap,
        H3_last_dropout)

print(np.average(JSD_H1_H3))
print(np.max(JSD_H1_H3))
print(np.min(JSD_H1_H3))

