import warnings
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.model_selection import StratifiedKFold
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
# import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px
import pandas as pd
import os


seed = 42
max_iter = 200

train_file_path = "records/train_em/mag1.npy"
test_file_path = "records/test_em/mag1.npy"
train_label_file_path = "records/train_label/mag1.npy"
test_label_file_path = "records/test_label/mag1.npy"


'''mag_claloss'''
train_embeddings = np.load(train_file_path)
train_embeddings_reshaped = train_embeddings.reshape(train_embeddings.shape[0], -1)
test_embeddings = np.load(test_file_path)
test_embeddings_reshaped = test_embeddings.reshape(test_embeddings.shape[0], -1)
train_labels = np.load(train_label_file_path)
train_labels = train_labels.astype(int)
test_labels = np.load(test_label_file_path)
test_labels = test_labels.astype(int)

clf = LinearSVC(random_state=seed, max_iter=max_iter)
clf.fit(train_embeddings, train_labels)
preds = clf.predict(test_embeddings)
macro = f1_score(test_labels, preds, average='macro')
micro = f1_score(test_labels, preds, average='micro')
print("Test Micro-F1:", micro)
print("Test Macro-F1:", macro)


