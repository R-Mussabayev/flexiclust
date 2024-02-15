import time
import numpy as np
import numba as nb
from collections import Counter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn import datasets
from sklearn import metrics
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import MinMaxScaler

def pairwise_similarity(X):
    n = len(X)
    out = np.empty((n,n))
    np.fill_diagonal(out,1)
    for i in range(n):
        for j in range(i+1,n):
            out[i,j] = np.intersect1d(X[i],X[j]).shape[0] / min(X[i].shape[0] ,X[j].shape[0]) #Jaccard similarity
            out[j,i] = out[i,j]
    return out
    
@nb.njit
def single_basis(dm, d, start):
    n = dm.shape[0]    
    out = np.array([start])
    C = np.delete(np.arange(n),start)
    while C.shape[0] > 0:
        C = C[np.sum(dm[out][:,C] <= d, axis=0) == len(out)]
        if C.shape[0] > 0:
            best = np.argsort(np.sum(dm[out][:,C], axis=0))[0]
            out = np.append(out,C[best])
            C = np.delete(C,best)
    return out
    
@nb.njit(parallel=True)
def all_bases(D, d):
    n = D.shape[0]
    inds = np.arange(n)
    out = [np.array([0])] * n
    for i in nb.prange(n):
        loc = inds[D[i] <= d]
        if loc.shape[0] > 0:
            basis = single_basis(D[loc][:,loc], d, np.where(loc == i)[0][0])
            if basis.shape[0] > 0:
                out[i] = loc[basis]
            else:
                out[i] = np.empty(0,dtype=nb.int64)
        else:
            out[i] = np.empty(0,dtype=nb.int64)
    return out

def unique_bases(bases, min_size = 1):
    u_bases = Counter([frozenset(basis) for basis in bases if (len(basis) > 0) and (len(basis) >= min_size)])
    unique = np.empty(len(u_bases), object)
    unique[:] = [np.array(list(basis)) for basis in list(u_bases.keys())]
    return unique

def merging(bases, pairwise, q):
    n = pairwise.shape[0]
    tril = np.tril(np.full((n,n),True),-1) # Lower triangle of matrix
    mask = pairwise >= q
    tril_and_mask = tril & mask
    
    graph = csr_matrix(tril_and_mask)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
        
    uniq_labels = np.unique(labels)
    n_uniq_labels = len(uniq_labels)
    bound_bases = np.empty(n_uniq_labels, object)
    bound_bases[:] = [np.array(list(set().union(*bases[labels == i])), dtype = int) for i in uniq_labels]    
    return labels, bound_bases

def flexi_clust(D, d, q, min_size = 1):
    bases = all_bases(D, d)   
    unique = unique_bases(bases, min_size)
    n_unique = len(unique)
    if n_unique > 0:
        weights = np.empty(n_unique, object)
        weights[:] = [np.mean(1-D[unique[i]][:,unique[i]], axis=0) for i in range(n_unique)]        
        sizes = np.array([np.size(unique[i]) for i in range(n_unique)])
        sizes = sizes / np.max(sizes)       
        weights = weights * sizes
        pairwise = pairwise_similarity(unique)
        structure, clusters = merging(unique, pairwise, q)
    else:
        weights = np.array([])
        clusters = np.array([], dtype = int)
        pairwise = np.array([])
        structure = np.array([], dtype = int)
    return unique, weights, structure, clusters, pairwise


def fuzzy_to_crisp(n, unique, weights, structure):
    m = len(unique)
    mat = np.zeros((m,n))
    for i in range(m):
        mat[i,unique[i]] = weights[i]
    return structure[np.argmax(mat.T, axis = 1)]


def accuracy(labels_true,labels_pred):
    return (metrics.adjusted_rand_score(labels_true, labels_pred)+1)/2


dataset = datasets.load_iris()
#dataset = datasets.load_wine() 
#dataset = datasets.load_breast_cancer()

X = MinMaxScaler().fit_transform(dataset.data)
Y = dataset.target

D = pairwise_distances(X, metric = 'euclidean', n_jobs = -1)


#Iris: d = 0.27069588, q = 0.40694871, Accuracy = 0.951
#Wine: d = 0.99826189, q = 0.74976414, Accuracy = 0.928
#Breast cancer: d = 0.81830551, q = 0.64369976, Accuracy = 0.826

d = 0.27069588
q = 0.40694871
min_size = 1

unique, weights, structure, clusters, pairwise = flexi_clust(D, d, q, min_size)

labels = fuzzy_to_crisp(len(X), unique, weights, structure)
print('Accuracy: ', accuracy(labels, Y))