from sklearn.metrics import silhouette_score
from scipy.cluster import hierarchy

def compute_silhouette_scores(lm, data_array, k_list, verbose=True):
    """Computes Silhouette scores for assessing clusters.
    
    lm: a linkage matrix
    data_array: the original array of features
    k_list: a list of integers (cluster numbers to consider)
    verbose: boolean
    """

    ss = []
    for k in k_list:
        if verbose:
            print(f'Computing groupings for k={k}')
        groupings = hierarchy.cut_tree(lm, n_clusters=k).ravel()
        if verbose:
            print(f'Computing score for k={k}')
        ss.append(silhouette_score(data_array, groupings))
    return(ss)