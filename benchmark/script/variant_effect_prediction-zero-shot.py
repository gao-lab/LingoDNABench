import numpy as np
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances

from sklearn.metrics import roc_auc_score
import os
import sys

# def cosine_similarity(vector_a, vector_b):
#     dot_product = np.dot(vector_a, vector_b)
#     norm_a = np.linalg.norm(vector_a)
#     norm_b = np.linalg.norm(vector_b)
#     if norm_a == 0 or norm_b == 0:
#         return 0
#     else:
#         return dot_product / (norm_a * norm_b)

def euclidean_distance(vector_a, vector_b):
    diff = vector_a - vector_b
    squared_diff = diff ** 2
    sum_squared_diff = np.sum(squared_diff)
    distance = np.sqrt(sum_squared_diff)
    return distance

dirnames=sys.argv[1]
model_name=sys.argv[2]
layer=int(sys.argv[3])

with open(f"{dirnames}/metrics-{model_name}-{layer}.txt",'w') as f1:
    label=np.concatenate([np.loadtxt(f"{dirnames}/{data_type}_label.txt",dtype=int) for data_type in ['test','dev','train']])
    
    data_ref=np.concatenate([np.load(f"{dirnames}/{model_name}/{data_type}_data_0-embedding-layer_{layer}.npy") for data_type in ['test','dev','train']])
    data_var=np.concatenate([np.load(f"{dirnames}/{model_name}/{data_type}_data_1-embedding-layer_{layer}.npy") for data_type in ['test','dev','train']])

    d_score=[]
    for i in range(len(data_ref)):
        d_score.append(euclidean_distance(data_ref[i],data_var[i]))
    d_score=np.array(d_score)

    to_save=np.array([i for i in range(len(d_score)) if not np.isnan(d_score[i])])
    d_score=d_score[to_save]
    label=label[to_save]
    d_auc=roc_auc_score(label,d_score)
    f1.write(f"{d_auc}\n")
    


