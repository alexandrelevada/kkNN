#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

KKNN: an adaptive curvature based nearest neighbor classifier

@author: Alexandre L. M. Levada

Python script to reproduce the results obtained by the first set of experiments in the paper

"""

# Imports
import os
import sys
import time
import warnings
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import sklearn.neighbors as sknn
import sklearn.utils.graph as sksp
from scipy import stats
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import top_k_accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import wilcoxon

# Gram-Schmidt ortogonalization
def gs(X, row_vecs=True, norm = True):
    if not row_vecs:
        X = X.T
    Y = X[0:1,:].copy()
    for i in range(1, X.shape[0]):
        proj = np.diag((X[i,:].dot(Y.T)/np.linalg.norm(Y,axis=1)**2).flat).dot(Y)
        Y = np.vstack((Y, X[i,:] - proj.sum(0)))
    if norm:
        Y = np.diag(1/np.linalg.norm(Y,axis=1)).dot(Y)
    if row_vecs:
        return Y
    else:
        return Y.T

# Computes the curvatures of all samples in the training set
def Curvature_Estimation(dados, k):
    n = dados.shape[0]
    m = dados.shape[1]
    # First fundamental form
    I = np.zeros((m, m))
    Squared = np.zeros((m, m))
    ncol = (m*(m-1))//2
    Cross = np.zeros((m, ncol))
    # Second fundamental form
    II = np.zeros((m, m))
    S = np.zeros((m, m))
    curvatures = np.zeros(n)
    # Generate KNN graph
    knnGraph = sknn.kneighbors_graph(dados, n_neighbors=k, mode='connectivity', include_self=False)
    A = knnGraph.toarray()    
    # Computes the means and covariance matrices for each patch
    for i in range(n):       
        vizinhos = A[i, :]
        indices = vizinhos.nonzero()[0]
        #  Computation of the first fundamental form
        amostras = dados[indices]
        ni = len(indices)
        if ni > 1:
            I = np.cov(amostras.T)
        else:
            I = np.eye(m)      # isolated points
        # Compute the eigenvectors
        v, w = np.linalg.eig(I)
        # Sort the eigenvalues
        ordem = v.argsort()
        # Select the eigenvectors in decreasing order (in columns)
        Wpca = w[:, ordem[::-1]]
        # Computation of the second fundamental form
        for j in range(0, m):
            Squared[:, j] = Wpca[:, j]**2
        col = 0
        for j in range(0, m):
            for l in range(j, m):
                if j != l:
                    Cross[:, col] = Wpca[:, j]*Wpca[:, l]
                    col += 1
        # Add a column of ones
        Wpca = np.column_stack((np.ones(m), Wpca))
        Wpca = np.hstack((Wpca, Squared))
        Wpca = np.hstack((Wpca, Cross))
        # Gram-Schmidt ortogonalization
        Q = gs(Wpca)
        # Discard the first m columns of H
        H = Q[:, (m+1):]        
        II = np.dot(H, H.T)
        S = -np.dot(II, I)
        curvatures[i] = abs(np.linalg.det(S))
    return curvatures

# Computes the curvature of a single point (test sample)
def Point_Curvature_Estimation(dados, k):
    n = dados.shape[0]
    m = dados.shape[1]
    # First fundamental form 
    I = np.zeros((m, m))
    Squared = np.zeros((m, m))
    ncol = (m*(m-1))//2
    Cross = np.zeros((m, ncol))
    # Second fundamental form
    II = np.zeros((m, m))
    S = np.zeros((m, m))
    curvature = 0
    amostras = dados
    ni = n
    # Computation of the first fundamental form
    I = np.cov(amostras.T)
    # Compute the eigenvectors
    v, w = np.linalg.eig(I)
    # Sort the eigenvalues
    ordem = v.argsort()
    # Select the eigenvectors in decreasing order (in columns)
    Wpca = w[:, ordem[::-1]]
    # Computation of the second fundamental form
    for j in range(0, m):
        Squared[:, j] = Wpca[:, j]**2
    col = 0
    for j in range(0, m):
        for l in range(j, m):
            if j != l:
                Cross[:, col] = Wpca[:, j]*Wpca[:, l]
                col += 1
    # Add a column of ones
    Wpca = np.column_stack((np.ones(m), Wpca))
    Wpca = np.hstack((Wpca, Squared))
    Wpca = np.hstack((Wpca, Cross))
    # Gram-Schmidt ortogonalization
    Q = gs(Wpca)
    # Discard the first m columns of H        
    H = Q[:, (m+1):]
    II = np.dot(H, H.T)
    S = -np.dot(II, I)
    curvature = abs(np.linalg.det(S))
    return curvature

# Optional function to normalize the curvatures to the interval [0, 1]
def normalize_curvatures(curv):
    k = (curv - curv.min())/(curv.max() - curv.min())
    return k

# Generates the k-NNG (fixed k)
def Simple_Graph(dados, k):
    n = dados.shape[0]
    m = dados.shape[1]
    # Generate k-NN graph
    knnGraph = sknn.kneighbors_graph(dados, n_neighbors=k, mode='distance', include_self=False)
    A = knnGraph.toarray()
    return A

# Generates the adaptive k-NNG (different k for each sample)
def Curvature_Based_Graph(dados, k, curv):
    n = dados.shape[0]
    m = dados.shape[1]
    # Generate KNN graph
    knnGraph = sknn.kneighbors_graph(dados, n_neighbors=k, mode='distance', include_self=False)
    A = knnGraph.toarray()
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(dados)
    distances, neighs = nbrs.kneighbors(dados)
    # Disconnect farthest samples
    for i in range(n):
        less = curv[i]
        A[i, neighs[nn-less:nn+1]] = 0
    return A

####################################
# Regular k-NN classifier
####################################

# Train regular k-NN classifier
def treina_KNN(treino, target, nn):
    # Just build the k-NNG
    A = Simple_Graph(dados, nn)
    return A    

# Test regular k-NN classifier
def testa_KNN(treino, teste, target_treino, target_teste, nn):
    n = teste.shape[0]
    m = teste.shape[1]
    labels = np.zeros(len(target_teste))
    for i in range(n):
        data = np.vstack((treino, teste[i, :]))
        rotulos = np.hstack((target_treino, target_teste[i]))
        knnGraph = sknn.kneighbors_graph(data, n_neighbors=nn, mode='distance', include_self=False)
        A = knnGraph.toarray()
        vizinhos = A[-1, :]                 # last line of the adjacency matrix
        indices = vizinhos.nonzero()[0]
        labels[i] = stats.mode(rotulos[indices])[0]
        del data
        del rotulos
    return labels

##############################################
# Adaptive curvature based kk-NN classifier
##############################################

# Train the adaptive kk-NN classifier
def treina_curvature_KNN(treino, target, nn):
    curvaturas = Curvature_Estimation(treino, nn)
    K = curvaturas
    intervalos = np.linspace(0.1, 0.9, 9)       # for curvature quantization quantization
    quantis = np.quantile(K, intervalos)
    bins = np.array(quantis)
    # Discrete curvature values obtained after quantization (scores)
    disc_curv = np.digitize(K, bins)
    A = Curvature_Based_Graph(treino, nn, disc_curv)
    return K

# Test the adaptive kk-NN classifier
def testa_curvature_KNN(treino, teste, target_treino, target_teste, nn):
    n = teste.shape[0]
    m = teste.shape[1]
    labels = np.zeros(len(target_teste))
    # Computes the curvature of the training set
    curvaturas = Curvature_Estimation(treino, nn)
    intervalos = np.linspace(0.1, 0.9, 9)       # for curvature quantization
    K = curvaturas
    quantis = np.quantile(K, intervalos)
    bins = np.array(quantis)
    # Discrete curvature values obtained after quantization (scores)
    disc_curv = np.digitize(K, bins)
    print()
    print('Size of test size: ', n)
    for i in range(n):
        # Computes the nearest neighbors of the i-th test sample
        nbrs = NearestNeighbors(n_neighbors=nn, algorithm='ball_tree').fit(treino)
        distances, neighs = nbrs.kneighbors(teste[i, :].reshape(1, -1))
        neighs = neighs[0]
        distances = distances[0]
        # Test sample + k nearest neighbors
        data = np.vstack((teste[i, :], treino[neighs, :]))  # add sample at the beginning
        curvature = Point_Curvature_Estimation(data, nn)
        # Add curvature in the vector of curvsatures
        curvaturas_ = np.hstack((K, curvature))
        quantis_ = np.quantile(curvaturas_, intervalos)
        bins_ = np.array(quantis_)
        disc_curv_ = np.digitize(curvaturas_, bins_)
        # Test sample + training set
        data_ = np.vstack((treino, teste[i, :]))  # add test sample at the end
        knnGraph = sknn.kneighbors_graph(data_, n_neighbors=nn, mode='distance', include_self=False)
        A = knnGraph.toarray()
        rotulos = np.hstack((target_treino, target_teste[i]))
        less = disc_curv_[-1]   # curvature score of the test sample
        ordem = A[-1, :].argsort()[::-1] 
        for j in range(0, less):
            #  We must assure to keep at least 1 nearest neighbor
            if len(np.nonzero(A[-1, :])[0]) > 1:
                A[-1, ordem[j]] = 0          
        vizinhos = A[-1, :]     # last row of the adjacency matrix
        indices = vizinhos.nonzero()[0]
        labels[i] = stats.mode(rotulos[indices])[0]        
        del data, data_, rotulos, curvaturas_, quantis_, bins_, disc_curv_
    return labels

# To avoid unnecessary warning messages
warnings.simplefilter(action='ignore')

############################################################
# Data loading (uncomment one dataset from the list below)
############################################################
X = skdata.fetch_openml(name='UMIST_Faces_Cropped', version=1)     
#X = skdata.fetch_openml(name='variousCancers_final', version=1)     
#X = skdata.fetch_openml(name='micro-mass', version=1)  
#X = skdata.fetch_openml(name='collins', version=2)                 

dados = X['data']
target = X['target']

# Convert labels to integers
if not isinstance(dados, np.ndarray):
    cat_cols = dados.select_dtypes(['category']).columns
    dados[cat_cols] = dados[cat_cols].apply(lambda x: x.cat.codes)
    # Convert to numpy
    dados = dados.to_numpy()
    le = LabelEncoder()
    le.fit(target)
    target = le.transform(target)

# Remove nan's
dados = np.nan_to_num(dados)

# Data standardization (to deal with variables having different units/scales)
dados = preprocessing.scale(dados)

n = dados.shape[0]
m = dados.shape[1]
# Number of neighbors
nn = round(np.log2(n))  
# Number of classes
c = len(np.unique(target))
# if even, add 1 to become odd
if nn % 2 == 0:
    nn += 1

print('N = ', n)
print('M = ', m)
print('C = %d' %c)
print('K = %d' %nn)
print()
input('Press enter to continue...')
print()

num_features = 10
if m > 100:
    print('Applying LDA to reduce the number of features...')
    print()
    model = LinearDiscriminantAnalysis(n_components=min(c-1, num_features))
    dados = model.fit_transform(dados, target)

# Size of the training sets
treino_sizes = [0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25]

matriz_knn = np.zeros((len(treino_sizes), 7))   # 7 performance evaluation metrics
matriz_kknn = np.zeros((len(treino_sizes), 7))  # 7 performance evaluation metrics

inicio = time.time()

for i, size in enumerate(treino_sizes):
    print('********************************************')
    print('********** Size of training set: %.2f' %size)
    print('********************************************')
    print()

    treino, teste, target_treino, target_teste = train_test_split(dados, target, train_size=size, random_state=42)

    # regular k-NN 
    rotulos_ = testa_KNN(treino, teste, target_treino, target_teste, nn)
    acc_ = accuracy_score(target_teste, rotulos_)
    bal_acc_ = balanced_accuracy_score(target_teste, rotulos_)
    f1_ = f1_score(target_teste, rotulos_, average='weighted')
    kappa_ = cohen_kappa_score(target_teste, rotulos_)
    prec_ = precision_score(target_teste, rotulos_, average='weighted')
    rec_ = recall_score(target_teste, rotulos_, average='weighted')
    jac_ = jaccard_score(target_teste, rotulos_, average='weighted')

    print('Regular KNN')
    print('-------------')
    print('Accuracy:', acc_)
    print('Balanced accuracy:', bal_acc_)
    print('F1 score:', f1_)
    print('Kappa:', kappa_)
    print('Precision:', prec_)
    print('Recall:', rec_)
    print('Jaccard:', jac_)

    # Adaptive curvature based kk-NN
    rotulos = testa_curvature_KNN(treino, teste, target_treino, target_teste, nn)
    acc = accuracy_score(target_teste, rotulos)
    bal_acc = balanced_accuracy_score(target_teste, rotulos)
    f1 = f1_score(target_teste, rotulos, average='weighted')
    kappa = cohen_kappa_score(target_teste, rotulos)
    prec = precision_score(target_teste, rotulos, average='weighted')
    rec = recall_score(target_teste, rotulos, average='weighted')
    jac = jaccard_score(target_teste, rotulos, average='weighted')

    print()
    print('Curvature based KNN')
    print('--------------------')
    print('Accuracy:', acc)
    print('Balanced accuracy:', bal_acc)
    print('F1 score:', f1)
    print('Kappa:', kappa)
    print('Precision:', prec)
    print('Recall:', rec)
    print('Jaccard:', jac)
    print()

    measures_knn = np.array([acc_, bal_acc_, f1_, kappa_, prec_, rec_, jac_])
    measures_curvature_knn = np.array([acc, bal_acc, f1, kappa, prec, rec, jac])

    matriz_knn[i, :] = measures_knn
    matriz_kknn[i, :] = measures_curvature_knn

fim = time.time()

print('Elapsed time : %f s' %(fim-inicio))
print()

print('Wilcoxon\'s test - Balanced Accuracy')
print(wilcoxon(matriz_knn[:, 1], matriz_kknn[:, 1]))
print()

print('Median balanced accuracy (regular k-NN):', np.median(matriz_knn[:, 1]))
print('Median balanced accuracy (kk-NN):', np.median(matriz_kknn[:, 1]))
print()
print('Median Kappa (regular k-NN):', np.median(matriz_knn[:, 3]))
print('Median Kappa (kk-NN):', np.median(matriz_kknn[:, 3]))
print()
print('Median Jaccard index (regular k-NN):', np.median(matriz_knn[:, 6]))
print('Median Jaccard index (kk-NN):', np.median(matriz_kknn[:, 6]))
print()
print('Median F1 score (regular k-NN):', np.median(matriz_knn[:, 2]))
print('Median F1 score (kk-NN):', np.median(matriz_kknn[:, 2]))

# Plota gráficos
if 'details' in X.keys():
    dataset_name = X['details']['name'] # apenas se for OpenML
else:
    dataset_name = 'digits'

num_sizes = len(treino_sizes)

# Folder to save the plots
if not os.path.exists('./performance'):
    os.mkdir('./performance')

if not os.path.exists('./performance/'+dataset_name):
    os.mkdir('./performance/'+dataset_name)

plt.figure(1)
plt.plot(treino_sizes[:num_sizes], matriz_knn[:num_sizes, 0], c='red', marker='*', label='k-NN')
plt.plot(treino_sizes[:num_sizes], matriz_kknn[:num_sizes, 0], c='blue', marker='*', label='kk-NN')
plt.xlabel('Training set sizes (percentages)')
plt.ylabel('Accuracy')
plt.title(dataset_name)
plt.legend()
plt.savefig('./performance/'+dataset_name+'/Accuracy.png')
plt.close()

plt.figure(2)
plt.plot(treino_sizes[:num_sizes], matriz_knn[:num_sizes, 1], c='red', marker='*', label='k-NN')
plt.plot(treino_sizes[:num_sizes], matriz_kknn[:num_sizes, 1], c='blue', marker='*', label='kk-NN')
plt.xlabel('Training set sizes (percentages)')
plt.ylabel('Balanced accuracy')
plt.title(dataset_name)
plt.legend()
plt.savefig('./performance/'+dataset_name+'/Balanced_Accuracy.png')
plt.close()

plt.figure(3)
plt.plot(treino_sizes[:num_sizes], matriz_knn[:num_sizes, 2], c='red', marker='*', label='k-NN')
plt.plot(treino_sizes[:num_sizes], matriz_kknn[:num_sizes, 2], c='blue', marker='*', label='kk-NN')
plt.xlabel('Training set sizes (percentages)')
plt.ylabel('F1 score')
plt.title(dataset_name)
plt.legend()
plt.savefig('./performance/'+dataset_name+'/F1_Score.png')
plt.close()

plt.figure(4)
plt.plot(treino_sizes[:num_sizes], matriz_knn[:num_sizes, 3], c='red', marker='*', label='k-NN')
plt.plot(treino_sizes[:num_sizes], matriz_kknn[:num_sizes, 3], c='blue', marker='*', label='kk-NN')
plt.xlabel('Training set sizes (percentages)')
plt.ylabel('Kappa coefficient')
plt.title(dataset_name)
plt.legend()
plt.savefig('./performance/'+dataset_name+'/Kappa.png')
plt.close()

plt.figure(5)
plt.plot(treino_sizes[:num_sizes], matriz_knn[:num_sizes, 4], c='red', marker='*', label='k-NN')
plt.plot(treino_sizes[:num_sizes], matriz_kknn[:num_sizes, 4], c='blue', marker='*', label='kk-NN')
plt.xlabel('Training set sizes (percentages)')
plt.ylabel('Precision')
plt.title(dataset_name)
plt.legend()
plt.savefig('./performance/'+dataset_name+'/Precision.png')
plt.close()

plt.figure(6)
plt.plot(treino_sizes[:num_sizes], matriz_knn[:num_sizes, 5], c='red', marker='*', label='k-NN')
plt.plot(treino_sizes[:num_sizes], matriz_kknn[:num_sizes, 5], c='blue', marker='*', label='kk-NN')
plt.xlabel('Training set sizes (percentages)')
plt.ylabel('Recall')
plt.title(dataset_name)
plt.legend()
plt.savefig('./performance/'+dataset_name+'/Recall.png')
plt.close()

plt.figure(7)
plt.plot(treino_sizes[:num_sizes], matriz_knn[:num_sizes, 6], c='red', marker='*', label='k-NN')
plt.plot(treino_sizes[:num_sizes], matriz_kknn[:num_sizes, 6], c='blue', marker='*', label='kk-NN')
plt.xlabel('Training set sizes (percentages)')
plt.ylabel('Jaccard index')
plt.title(dataset_name)
plt.legend()
plt.savefig('./performance/'+dataset_name+'/Jaccard.png')
plt.close()