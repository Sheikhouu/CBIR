import numpy as np
from scipy.spatial import distance

def manhattan(v1, v2):
    v1 = np.array(v1).astype('float')
    v2 = np.array(v2).astype('float')
    dist = np.sum(np.abs(v1 - v2))
    return dist

def euclidean(v1, v2):
    v1 = np.array(v1).astype('float')
    v2 = np.array(v2).astype('float')
    dist = np.sqrt(np.sum((v1 - v2) ** 2))
    return dist

def chebyshev(v1, v2):
    v1 = np.array(v1).astype('float')
    v2 = np.array(v2).astype('float')
    dist = np.max(np.abs(v1 - v2))
    return dist

def canberra(v1, v2):
    return distance.canberra(v1, v2)

def retrieve_similar_images(signature_db, query_features, distance_measure, num_results):
    distances = []
    for instance in signature_db:
        signature_features, label, img_path = instance[:-2], instance[-2], instance[-1]
        if distance_measure == 'manhattan':
            dist = manhattan(query_features, signature_features)
        elif distance_measure == 'euclidean':
            dist = euclidean(query_features, signature_features)
        elif distance_measure == 'chebyshev':
            dist = chebyshev(query_features, signature_features)
        elif distance_measure == 'canberra':
            dist = canberra(query_features, signature_features)
        distances.append((img_path, dist, label))
    distances.sort(key=lambda x: x[1])
    return distances[:num_results]
