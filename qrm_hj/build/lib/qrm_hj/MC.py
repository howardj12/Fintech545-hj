import pandas as pd
import numpy as np

#Calculate the exponentially weighted covariance matrix
def expo_weighted_cov(ret_data,w_lambda):
    weight = np.zeros(len(ret_data))
    for i in range(len(ret_data)):
        weight[len(ret_data)-1-i]  = (1-w_lambda)*w_lambda**i
    weight = weight/sum(weight)
    ret_means = ret_data - ret_data.mean()
    expo_w_cov = ret_means.T.values @ np.diag(weight) @ ret_means.values
    return expo_w_cov

# Pearson corr mtx + EW var vec
def PS_corr_mtx_EW_var_vec(ret_data, w_lambda=0.97):
    ew_cov_mtx = expo_weighted_cov(ret_data, w_lambda)
    #np.diag(np.reciprocal(np.sqrt(np.diag(ew_cov_mtx)))) 
    std_dev = np.sqrt(np.diag(ew_cov_mtx))
    corr = np.corrcoef(ret_data.T)
    return np.diag(std_dev) @ corr @ np.diag(std_dev).T

#EW corr mtx + PS var vec
def EW_corr_mtx_PS_var_vec(ret_data, w_lambda=0.97):
    ew_cov_mtx = expo_weighted_cov(ret_data, w_lambda)

    invSD = np.diag(np.reciprocal(np.sqrt(np.diag(ew_cov_mtx))))
    corr = invSD.dot(ew_cov_mtx).dot(invSD)

    var = np.var(ret_data)
    std_dev = np.sqrt(var)
    return np.diag(std_dev) @ corr @ np.diag(std_dev).T
####################################################################

# Rebonato and Jackel deal with non-PSD matrix for correlation mtx
def near_psd(mtx, epsilon=0.0):
    n = mtx.shape[0]

    invSD = None
    out = mtx.copy()

    # # calculate the correlation matrix if we got a covariance
    # if (np.diag(out) == 1.0).sum() != n:
    #     invSD = np.diag(1 / np.sqrt(np.diag(out)))
    #     out = invSD.dot(out).dot(invSD)

    # SVD, update the eigen value and scale
    vals, vecs = np.linalg.eigh(out)
    vals = np.maximum(vals, epsilon)
    T = np.reciprocal(np.square(vecs).dot(vals))
    T = np.diag(np.sqrt(T))
    l = np.diag(np.sqrt(vals))
    B = T.dot(vecs).dot(l)
    out = np.matmul(B, np.transpose(B))

    # # Add back the variance
    # if invSD is not None:
    #     invSD = np.diag(1 / np.diag(invSD))
    #     out = invSD.dot(out).dot(invSD)

    return out

#Higham deal with non-PSD matrix for correlation mtx
#First projection
def Pu(mtx):
    new_mtx = mtx.copy()
    for i in range(len(mtx)):
        for j in range(len(mtx[0])):
            if i == j:
                new_mtx[i][j] = 1
    return new_mtx
#Second projection
def Ps(mtx, w):
    mtx = np.sqrt(w)@mtx@np.sqrt(w)
    vals, vecs = np.linalg.eigh(mtx)
    vals = np.array([max(i,0) for i in vals])
    new_mtx = np.sqrt(w)@ vecs @ np.diag(vals) @ vecs.T @ np.sqrt(w)
    return new_mtx

#Calculate Frobenius Norm
def fnorm(mtxa, mtxb):
    s = mtxa - mtxb
    norm = 0
    for i in range(len(s)):
        for j in range(len(s[0])):
            norm +=s[i][j]**2
    return norm


def higham_psd(mtx, w, max_iteration = 1000, tol = 1e-8):
    r0 = np.inf
    Y = mtx
    S = np.zeros_like(Y)
 
    invSD = None
    # if np.count_nonzero(np.diag(Y) == 1.0) != mtx.shape[0]:
    #     invSD = np.diag(1.0 / np.sqrt(np.diag(Y)))
    #     Y = invSD.dot(Y).dot(invSD)
    C = Y.copy()
    
    for i in range(max_iteration):
        R = Y - S
        X = Ps(R, w)
        S = X - R
        Y = Pu(X)
        r = fnorm(Y, C)
        minval = np.linalg.eigvals(Y).min()
        if abs(r - r0) < tol and minval > -1e-8:
            break
        else:
            r0 = r
    
    # if invSD is not None:
    #     invSD = np.diag(1 / np.diag(invSD))
    #     Y =invSD.dot(Y).dot(invSD)
    return Y

#Confirm matrix is PSD or not
def psd(mtx):
    eigenvalues = np.linalg.eigh(mtx)[0]
    return np.all(eigenvalues >= -1e-8)
###################################################################

#Multivariate normal distribution
#Cholesky Factorization for PSD matrix
def chol_psd(cov_matrix):

    cov_mtx = cov_matrix
    n = cov_mtx.shape[0]
    root = np.zeros_like(cov_mtx)
    for j in range(n):
        s = 0.0
        if j > 0:
            # calculate dot product of the preceeding row values
            s = np.dot(root[j, :j], root[j, :j])
        temp = cov_mtx[j, j] - s
        if 0 >= temp >= -1e-8:
            temp = 0.0
        root[j, j] = np.sqrt(temp)
        if root[j, j] == 0.0:
            # set the column to 0 if we have an eigenvalue of 0
            root[j + 1:, j] = 0.0
        else:
            ir = 1.0 / root[j, j]
            for i in range(j + 1, n):
                s = np.dot(root[i, :j], root[j, :j])
                root[i, j] = (cov_mtx[i, j] - s) * ir
    return root


#Direct simulation
def multiv_normal_sim(cov_mtx, n_draws):
    L = chol_psd(cov_mtx)
    std_normals = np.random.normal(size=(len(cov_mtx), n_draws))
    return np.transpose((L @ std_normals) + 0)

#PCA simulation
def pca_sim(cov_mtx, n_draws, percent_explain):
    eigenvalues, eigenvectors = np.linalg.eig(cov_mtx)
    #Keep those positive eigenvalues and corresponding eigenvectors
    p_idx = eigenvalues > 1e-8
    eigenvalues = eigenvalues[p_idx]
    eigenvectors = eigenvectors[:, p_idx]
    #Sort
    s_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[s_idx]
    eigenvectors = eigenvectors[:, s_idx]

    if percent_explain == 1.0:
        percent_explain = (np.cumsum(eigenvalues)/np.sum(eigenvalues))[-1]

    n_eigenvalues = np.where((np.cumsum(eigenvalues)/np.sum(eigenvalues))>= percent_explain)[0][0] + 1
    #print(n_eigenvalues)
    eigenvectors = eigenvectors[:,:n_eigenvalues]
    eigenvalues = eigenvalues[:n_eigenvalues]
    std_normals = np.random.normal(size=(n_eigenvalues, n_draws))

    B = eigenvectors.dot(np.diag(np.sqrt(eigenvalues)))
    return np.transpose(B.dot(std_normals))
