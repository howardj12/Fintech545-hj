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


def pearson_corr_with_ew_variance(ret_data, w_lambda=0.97):
    ew_cov_mtx = expo_weighted_cov(ret_data, w_lambda)
    #np.diag(np.reciprocal(np.sqrt(np.diag(ew_cov_mtx)))) 
    std_dev = np.sqrt(np.diag(ew_cov_mtx))
    corr = np.corrcoef(ret_data.T)
    return np.diag(std_dev) @ corr @ np.diag(std_dev).T

def ew_corr_with_pearson_variance(ret_data, w_lambda=0.97):
    ew_cov_mtx = expo_weighted_cov(ret_data, w_lambda)

    invSD = np.diag(np.reciprocal(np.sqrt(np.diag(ew_cov_mtx))))
    corr = invSD.dot(ew_cov_mtx).dot(invSD)

    var = np.var(ret_data)
    std_dev = np.sqrt(var)
    return np.diag(std_dev) @ corr @ np.diag(std_dev).T

def near_psd(mat, epsilon=0.0):
    n = mat.shape[0]
    result = mat.copy()

    # Check if the input is a covariance matrix, and if so, convert to a correlation matrix
    if not np.allclose(np.diag(result), 1.0):
        scale = np.diag(1.0 / np.sqrt(np.diag(result)))
        result = scale @ result @ scale

    # Perform eigenvalue decomposition and adjust the eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(result)
    adjusted_eigenvalues = np.maximum(eigenvalues, epsilon)

    # Reconstruct the positive semi-definite matrix
    result = eigenvectors @ np.diag(np.sqrt(adjusted_eigenvalues)) @ eigenvectors.T

    # If the input was a covariance matrix, scale back to the original scale
    if 'scale' in locals():
        result = np.linalg.inv(scale) @ result @ np.linalg.inv(scale)
    
    return result

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


def higham_psd(mat, max_iterations=1000, tolerance=1e-8):
    Y = mat.copy()
    delta_s = np.zeros_like(mat)

    if not np.allclose(np.diag(Y), 1.0):
        inv_sd = np.diag(1.0 / np.sqrt(np.diag(Y)))
        Y = inv_sd @ Y @ inv_sd

    Y_prev = Y.copy()
    
    for i in range(max_iterations):
        R = Y - delta_s
        eigenvalues, eigenvectors = np.linalg.eigh(R)
        eigenvalues = np.maximum(eigenvalues, 0)
        X = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        delta_s = X - R
        Y = X.copy()
        np.fill_diagonal(Y, 1)

        # Check for convergence using the Frobenius norm of the difference between consecutive Y matrices
        if np.linalg.norm(Y - Y_prev, 'fro') < tolerance:
            break

        Y_prev = Y.copy()

    if 'inv_sd' in locals():
        Y = np.linalg.inv(inv_sd) @ Y @ np.linalg.inv(inv_sd)

    return Y

#Confirm matrix is PSD or not
def psd(mtx):
    eigenvalues = np.linalg.eigh(mtx)[0]
    return np.all(eigenvalues >= -1e-8)

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

##############################

## simulations method (e.g PCA)
# PCA to simulate the system through defining a new function
def simulate_PCA(a, nsim, percent_explained=1):
    # calculate the eigenvalues and eigenvectors of derived matrix, and sort eigenvalues from largest to smallest
    e_val, e_vec = eigh(a)
    sort_index = np.argsort(-1 * e_val)
    d_sorted_e_val = e_val[sort_index]
    d_sorted_e_vec = e_vec[:,sort_index]

    # we assume all negative eigenvalues derived are zero, since they are effectively zero (larger than -1e-8)
    assert np.amin(d_sorted_e_val) > -1e-8
    d_sorted_e_val[d_sorted_e_val<0] = 0
    
    # calculate the sum of all eigenvalues
    e_sum = sum(d_sorted_e_val)

    # choose a certain number of eigenvalues from the descending list of all eigenvalues so that the system explains the same percent of variance as the level inputed as parameter "percent_explained"
    total_percent = []
    sum_percent = 0
    for i in range(len(d_sorted_e_val)):
        each_percent = d_sorted_e_val[i] / e_sum
        sum_percent += each_percent
        total_percent.append(sum_percent)
    total_percent_np = np.array(total_percent)
    diff = total_percent_np - percent_explained
    abs_diff = abs(diff)
    index = np.where(abs_diff==abs_diff.min())
    
    # update eigenvalues and eigenvectors with the list of indices we generate above
    upd_e_val = d_sorted_e_val[:(index[0][0]+1)]
    upd_e_vec = d_sorted_e_vec[:,:(index[0][0]+1)]
    
    # construct the matrix for the simulating process
    B = upd_e_vec @ np.diag(np.sqrt(upd_e_val))
    r = np.random.randn(len(upd_e_val),nsim)
    
    result = B @ r
    result_t = np.transpose(result)
    
    return result_t

# direct simulation
def direct_simulate(a, nsim):
    # get eigenvalues and eigenvectors of the input matrix
    val, vec = eigh(a)
    sort_index = np.argsort(-1 * val)
    d_sorted_val = val[sort_index]
    d_sorted_vec = vec[:,sort_index]
    
    # to check if all eigenvalues are non-negative or negative but effectively zero, and set all effectively-zero eigenvalues to zero
    assert np.amin(d_sorted_val) > -1e-8
    d_sorted_val[d_sorted_val<0] = 0
    
    # construct the matrix for the simulating process
    B = d_sorted_vec @ np.diag(np.sqrt(d_sorted_val))
    r = np.random.randn(len(d_sorted_val),nsim)
    
    result = B @ r
    result_t = np.transpose(result)
    
    return result_t

# AR(1) simulation
def AR1_simulate(data,n=10000,ahead=1):
    params = fit_AR1(data)
    l = len(data)
    out = pd.DataFrame(0,index=range(ahead), columns=range(n))
    data_last = data[l-1] - params["Mean"]
    
    for i in range(n):
        datal = data_last
        next = 0
        for j in range(ahead):
            next = params["Coefficient"] * datal + params["sqrt_Sig2"] * np.random.normal()
            datal = next
            out[j,i] = next
    
    out = out + params["Mean"]
    return out

##############################

## fitting data into distributions
# define a function to fit data into several distributions using MLE (either normal distribution or t distribution) (note that log likelihood returned from this function is negated)
def fit_MLE(x,y,dist="norm"):
    # define another function to use MLE to fit data into normal distribution
    def LLfunc_nor(parameters, x, y): 
        # setting parameters
        beta1 = parameters[0] 
        beta0 = parameters[1] 
        sigma = parameters[2] 
    
        # derive estimated values of y
        y_est = beta1 * x + beta0 
    
        # compute log likelihood, but return the negative LL for convenience of later optimization
        LL = np.sum(norm.logpdf(y-y_est, loc=0, scale=sigma))
        return -LL
    # define another function to use MLE to fit data into t distribution
    def LLfunc_t(parameters, x, y): 
        # setting parameters
        beta1 = parameters[0] 
        beta0 = parameters[1] 
        sigma = parameters[2]
        df = parameters[3]
    
        # derive estimated values of y
        y_est = beta1 * x + beta0 
    
        # compute log likelihood, but return the negative LL for convenience of later optimization
        LL = np.sum(t.logpdf(y-y_est, sigma, df))
        return -LL
    # define another function to explicitly show constraints of our optimization problem
    def constraints(parameters):
        sigma = parameters[2]
        return sigma

    cons = {
        'type': 'ineq',
        'fun': constraints
    }
    
    if dist == "norm":
        lik_normal = minimize(LLfunc_nor, np.array([2, 2, 2]), args=(x,y))
        return {"Log_likelihood" : lik_normal.fun, "Parameters" : lik_normal.x}
    elif dist == "t":
        lik_t = minimize(LLfunc_t, np.array([1, 1, 1, 1]), args=(x,y))
        return {"Log_likelihood" : lik_t.fun, "Parameters" : lik_t.x}
    
# define a function to use linear regression to fit data
def linear_reg(x,y):
    reg = LinearRegression().fit(x,y)
    return {"Coefficients" : reg.coef_, "Intercept" : reg.intercept_}

# define a function to fit data into a AR(x) model, simulate data with the funtion - y1 = a1*y0 + s*np.random.normal + m (注意y0在一开始就要先减m）
def fit_AR1(data):
    mod = sm.tsa.ARIMA(data, order=(1, 0, 0))
    results = mod.fit()
    summary = results.summary()
    m = float(summary.tables[1].data[1][1])
    a1 = float(summary.tables[1].data[2][1])
    s = np.sqrt(float(summary.tables[1].data[3][1]))
    return {"Mean" : m, "Coefficient" : a1, "sqrt_Sig2" : s}

##############################

## calculation of covariance matrix
# basic transformation from correlation matrix to covariance matrix
def cor2cov(corel,vols):
    covar = np.diag(vols).dot(corel).dot(np.diag(vols))
    return covar

# basic transformation from covariance matrix to correlation matrix
def cov2cor(cov):
    std = np.sqrt(np.diag(cov))
    for i in range(len(cov.columns)):
        for j in range(len(cov.index)):
            cov.iloc[j,i] = cov.iloc[j,i] / (std[i] * std[j])
    return cov

# define a function to calculate exponential weights
def populateWeights(x,w,cw, λ):
    n = len(x)
    tw = 0
    # start a for loop to calculate the weight for each stock, and recording total weights and cumulative weights for each stock
    for i in range(n):
        individual_w = (1-λ)*pow(λ,i)
        w.append(individual_w)
        tw += individual_w
        cw.append(tw)
    
    # start another for loop to calculate normalized weights and normalized cumulative weights for each stock
    for i in range(n):
        w[i] = w[i]/tw
        cw[i] = cw[i]/tw

# define a function to calculate the exponentially weighted covariance matrix
def exwCovMat(data, weights_vector):
    # get the stock names listed in the file, and delete the first item, since it is the column of dates
    stock_names = list(data.columns)
    
    # set up an empty matrix, and transform it into a pandas Dataframe
    mat = np.empty((len(stock_names),len(stock_names)))
    w_cov_mat = pd.DataFrame(mat, columns = stock_names, index = stock_names)
    
    # calculate variances and covariances
    for i in stock_names:
        for j in stock_names:
            # get data of stock i and data of stock j respectively
            i_data = data.loc[:,i]
            j_data = data.loc[:,j]
            
            # calculate means of data of stock i and data of stock j
            i_mean = i_data.mean()
            j_mean = j_data.mean()
            
            # make sure i_data, j_data, and weights_vector all have the same number of items
            assert len(i_data) == len(j_data) == len(weights_vector)
            
            # set up sum for calculation of variance and covariance, and a for loop for that
            s = 0
            
            for z in range(len(data)):                
                part = weights_vector[z] * (i_data[z] - i_mean) * (j_data[z] - j_mean)
                s += part
            
            # store the derived variance into the matrix
            w_cov_mat.loc[i,j] = s
    
    return w_cov_mat

##############################

#####fix non-psd matrices#####START
# cholesky factorization for psd matrices
def chol_psd_forpsd(root,a):
    n = a.shape
    # initialize root matrix with zeros
    root = np.zeros(n)
    
    for j in range(n[0]):
        s = 0
        # if we are not on the first column, calculate the dot product of the preceeding row values
        if j > 1:
            s = np.dot(root[j,:(j-1)], root[j,:(j-1)])
            
        # working on diagonal elements
        temp = a[j,j] - s
        if temp <= 0 and temp >= -1e-8:
            temp = 0
        root[j,j] = np.sqrt(temp)
        
        # check for zero eigenvalues; set columns to zero if we have one(s)
        if root[j,j] == 0:
            root[j,(j+1):] = 0
        else:
            ir = 1/root[j,j]
            for i in range((j+1),n[0]):
                s = np.dot(root[i,:(j-1)], root[j,:(j-1)])
                root[i,j] = (a[i,j] - s) * ir
    return root

# cholesky factorization for pd matrices                
def chol_psd_forpd(root,a):
    n = a.shape
    # initialize root matrix with zeros
    root = np.zeros(n)
    
    for j in range(n[0]):
        s = 0
        # if we are not on the first column, calculate the dot product of the preceeding row values
        if j > 1:
            s = np.dot(root[j,:(j-1)], root[j,:(j-1)])
            
        # working on diagonal elements
        temp = a[j,j] - s
        root[j,j] = np.sqrt(temp)
        
        ir = 1/root[j,j]
        # update off diagonal rows of the column
        for i in range((j+1),n[0]):
            s = np.dot(root[i,:(j-1)], root[j,:(j-1)])
            root[i,j] = (a[i,j] - s) * ir
    return root            

## norm calculation
def F_Norm(M):
    # get the number of rows and columns of the input matrix M
    size = M.shape
    rows = size[0]
    columns = size[1]
    
    # compute the norm
    sum = 0
    for i in range(rows):
        for j in range(columns):
            square = pow(M[i][j],2)
            sum += square
    
    return np.sqrt(sum)
