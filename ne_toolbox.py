from scipy.stats import zscore
from sklearn.decomposition import PCA, FastICA
import numpy as np


def detect_cell_assemblies(spktrain):
    spktrain_z = zscore(spktrain, 1)
    pca = PCA()
    pca.fit(np.transpose(spktrain_z))

    # get threshold for significant PCs (based on Marcenko-Patur distribution)
    q = spktrain.shape[1] / spktrain.shape[0]
    thresh = (1 + np.sqrt(1 / q)) ** 2
    num_ne = sum(pca.explained_variance_ > thresh)
    patterns = fast_ica(spktrain_z, num_ne)
    return normalize_pattern(patterns)


def fast_ica(spktrain_z, num_ne, niter=500):
    """get cNE patterns using ICA"""
    # step1: PCA
    pca = PCA()
    pca.fit(np.transpose(spktrain_z))
    # get subspace spanned by num_ne PCa
    eigen_vectors = pca.components_[:num_ne]
    eigen_values = pca.explained_variance_[:num_ne]
    whiten_mat = np.diag(1 / np.sqrt(eigen_values)) @ eigen_vectors
    dewhiten_mat = np.transpose(eigen_vectors) @ np.diag(np.sqrt(eigen_values))
    spktrain_w = whiten_mat @ spktrain_z  # project spktrain_z to first n PCs and scale variance to 1
    # step2: ICA
    ica = FastICA(tol=1e-10, max_iter=niter, random_state=0, whiten=False)
    ica.fit(np.transpose(spktrain_w))
    unmixing = ica._unmixing
    return unmixing @ np.transpose(dewhiten_mat)


def normalize_pattern(patterns):
    """Normalize length of the patterns to be 1 and make the highest absolute deviation to be positive"""
    patterns = [x / np.sqrt(np.sum(x ** 2)) for x in patterns]
    patterns = [-x if x[abs(x).argmax()] < 0 else x for x in patterns]
    return np.array(patterns)
