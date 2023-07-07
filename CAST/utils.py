import networkx as nx
import numpy as np
import scipy, random
import scanpy as sc
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_chunked
from .visualize import link_plot

def coords2adjacentmat(coords,output_mode = 'adjacent',strategy_t = 'convex'):
    if strategy_t == 'convex': ### slow but may generate more reasonable delaunay graph
        from libpysal.cg import voronoi_frames
        from libpysal import weights
        cells, _ = voronoi_frames(coords, clip="convex hull")
        delaunay_graph = weights.Rook.from_dataframe(cells).to_networkx()
    elif strategy_t == 'delaunay': ### fast but may generate long distance edges
        from scipy.spatial import Delaunay
        from collections import defaultdict
        tri = Delaunay(coords)
        delaunay_graph = nx.Graph()
        coords_dict = defaultdict(list)
        for i, coord in enumerate(coords):
            coords_dict[tuple(coord)].append(i)
        for simplex in tri.simplices:
            for i in range(3):
                for node1 in coords_dict[tuple(coords[simplex[i]])]:
                    for node2 in coords_dict[tuple(coords[simplex[(i+1)%3]])]:
                        if not delaunay_graph.has_edge(node1, node2):
                            delaunay_graph.add_edge(node1, node2)
    if output_mode == 'adjacent':
        return nx.to_scipy_sparse_array(delaunay_graph).todense()
    elif output_mode == 'raw':
        return delaunay_graph
    elif output_mode == 'adjacent_sparse':
        return nx.to_scipy_sparse_array(delaunay_graph)

def hv_cutoff(max_col,threshold=2000):
    for thres_t in range(0,int(np.max(max_col))):
        if np.sum(max_col > thres_t) < threshold:
            return thres_t -1

def detect_highly_variable_genes(sdata,batch_key = 'batch',n_top_genes = 4000,count_layer = 'count'):
    samples = np.unique(sdata.obs[batch_key])
    thres_list = []
    max_count_list = []
    bool_list = []
    for list_t, sample_t in enumerate(samples):
        idx_t = sdata.obs[batch_key] == sample_t
        if count_layer == '.X':
            max_t = sdata[idx_t,:].X.max(0).toarray() if scipy.sparse.issparse(sdata.X) else sdata[idx_t,:].X.max(0)
        else:
            max_t = sdata[idx_t,:].layers[count_layer].max(0).toarray() if scipy.sparse.issparse(sdata.layers[count_layer]) else sdata[idx_t,:].layers[count_layer].max(0)
        max_count_list.append(max_t)
        thres_list.append(hv_cutoff(max_count_list[list_t],threshold=n_top_genes))
        bool_list.append(max_count_list[list_t] > thres_list[list_t])
    stack = np.stack(bool_list)
    return np.all(stack, axis=0)[0]

def extract_coords_exp(sdata, batch_key = 'batch', cols = 'spatial', count_layer = 'count', data_format = 'norm1e4',ifcombat = False, if_inte = False):
    coords_raw = {}
    exps = {}
    samples = np.unique(sdata.obs[batch_key])
    if count_layer == '.X':
        sdata.layers['raw'] = sdata.X.copy()
    sdata = preprocess_fast(sdata, mode = 'customized')
    if if_inte:
        scaled_layer = 'log2_norm1e4_scaled'
        pc_feature = 'X_pca_harmony'
        sdata = Harmony_integration(sdata,
                                    scaled_layer = scaled_layer,
                                    use_highly_variable_t = True,
                                    batch_key = batch_key,
                                    umap_n_neighbors = 50,
                                    umap_n_pcs = 30,
                                    min_dist = 0.01,
                                    spread_t = 5,
                                    source_sample_ctype_col = None,
                                    output_path = None,
                                    n_components = 50,
                                    ifplot = False,
                                    ifcombat = ifcombat)
        for sample_t in samples:
            idx_t = sdata.obs[batch_key] == sample_t
            coords_raw[sample_t] = sdata.obsm['spatial'][idx_t] if type(cols) is not list else np.array(sdata.obs[cols][idx_t])
            exps[sample_t] = sdata[idx_t].obsm[pc_feature].copy()
    else:
        sdata.X = sdata.layers[data_format].copy()
        if ifcombat == True:
            sc.pp.combat(sdata, key=batch_key)
        for sample_t in samples:
            idx_t = sdata.obs[batch_key] == sample_t
            coords_raw[sample_t] = sdata.obsm['spatial'][idx_t] if type(cols) is not list else np.array(sdata.obs[cols][idx_t])
            exps[sample_t] = sdata[idx_t].X.copy()
            if scipy.sparse.issparse(exps[sample_t]):
                exps[sample_t] = exps[sample_t].toarray()
    return coords_raw,exps

def Harmony_integration(
    sdata_inte,
    scaled_layer,
    use_highly_variable_t,
    batch_key,
    umap_n_neighbors,
    umap_n_pcs,
    min_dist,
    spread_t,
    source_sample_ctype_col,
    output_path,
    n_components = 50,
    ifplot = True,
    ifcombat = False):
    #### integration based on the Harmony
    sdata_inte.X = sdata_inte.layers[scaled_layer].copy()
    if ifcombat == True:
        sc.pp.combat(sdata_inte, key=batch_key)
    print(f'Running PCA based on the layer {scaled_layer}:')
    sc.tl.pca(sdata_inte, use_highly_variable=use_highly_variable_t, svd_solver = 'full', n_comps= n_components)
    print(f'Running Harmony integration:')
    sc.external.pp.harmony_integrate(sdata_inte, batch_key)
    print(f'Compute a neighborhood graph based on the {umap_n_neighbors} `n_neighbors`, {umap_n_pcs} `n_pcs`:')
    sc.pp.neighbors(sdata_inte, n_neighbors=umap_n_neighbors, n_pcs=umap_n_pcs, use_rep='X_pca_harmony')
    print(f'Generate the UMAP based on the {min_dist} `min_dist`, {spread_t} `spread`:')
    sc.tl.umap(sdata_inte,min_dist=min_dist, spread = spread_t)
    sdata_inte.obsm['har_X_umap'] = sdata_inte.obsm['X_umap'].copy()
    if ifplot == True:
        plt.rcParams.update({'pdf.fonttype':42})
        sc.settings.figdir = output_path
        sc.set_figure_params(figsize=(10, 10),facecolor='white',vector_friendly=True, dpi_save=300,fontsize = 25)
        sc.pl.umap(sdata_inte,color=[batch_key],size=10,save=f'_har_{umap_n_pcs}pcs_batch.pdf')
        sc.pl.umap(sdata_inte,color=[source_sample_ctype_col],size=10,save=f'_har_{umap_n_pcs}pcs_ctype.pdf') if source_sample_ctype_col is not None else None
    return sdata_inte

def sub_node_sum(coords_t,exp_t,nodenum=1000,vis = True,seed_t = 2):
    from scipy.sparse import csr_matrix as csr
    random.seed(seed_t)
    if nodenum > coords_t.shape[0]:
        print('The number of nodes is larger than the total number of nodes. Return the original data.')
        sub_node_idx = np.arange(coords_t.shape[0])
        if scipy.sparse.issparse(exp_t):
            return exp_t,sub_node_idx
        else:
            return csr(exp_t),sub_node_idx
    sub_node_idx = np.sort(random.sample(range(coords_t.shape[0]),nodenum))
    coords_t_sub = coords_t[sub_node_idx,:].copy()
    close_idx = nearest_neighbors_idx(coords_t_sub,coords_t)
    A = np.zeros([coords_t_sub.shape[0],coords_t.shape[0]])
    for ind,i in enumerate(close_idx.tolist()):
        A[i,ind] = 1
    csr_A = csr(A)
    if scipy.sparse.issparse(exp_t):
        exp_t_sub = csr_A.dot(exp_t)
    else:
        exp_t_sub = csr_A.dot(csr(exp_t))
    if(vis == True):
        link_plot(close_idx,coords_t,coords_t_sub,k = 1)
    return exp_t_sub,sub_node_idx

def nearest_neighbors_idx(coord1,coord2,mode_t = 'knn'): ### coord1 is the reference, coord2 is the target
    if mode_t == 'knn':
        from sklearn.neighbors import KNeighborsClassifier
        knn_classifier = KNeighborsClassifier(n_neighbors=1,metric='euclidean')
        knn_classifier.fit(coord1, np.zeros(coord1.shape[0]))  # Use dummy labels, since we only care about distances
        # Find nearest neighbors
        _, close_idx = knn_classifier.kneighbors(coord2)
        return close_idx
    else:
        result = []
        dists = pairwise_distances_chunked(coord2,coord1,working_memory = 100, metric='euclidean', n_jobs=-1)
        for chunk in tqdm(dists): # for each chunk (minibatch)
            knn_ind = np.argpartition(chunk, 0, axis=-1)[:, 0] # introsort to get indices of top k neighbors according to the distance matrix [n_query, k]
            result.append(knn_ind)
        close_idx = np.concatenate(result)
        return np.expand_dims(close_idx, axis=1)
    
def non_zero_center_scale(sdata_t_X):
    std_nocenter = np.sqrt(np.square(sdata_t_X).sum(0)/(sdata_t_X.shape[0]-1))
    return(sdata_t_X/std_nocenter)

def sub_data_extract(sample_list,coords_raw, exps, nodenum_t = 20000):
    coords_sub = dict()
    exp_sub = dict()
    sub_node_idxs = dict()
    for sample_t in sample_list:
        exp_t,sub_node_idxs[sample_t] = sub_node_sum(coords_raw[sample_t],exps[sample_t],nodenum=nodenum_t,vis = False)
        exp_sub[sample_t] = non_zero_center_scale(exp_t.toarray())
        coords_sub[sample_t] = coords_raw[sample_t][sub_node_idxs[sample_t],:]
    return coords_sub,exp_sub,sub_node_idxs

def preprocess_fast(sdata1, mode = 'customized',target_sum=1e4,base = 2,zero_center = True,regressout = False):
    print('Preprocessing...')
    from scipy.sparse import csr_matrix as csr
    if 'raw' in sdata1.layers:
        if type(sdata1.layers['raw']) != scipy.sparse._csr.csr_matrix:
            sdata1.layers['raw'] = csr(sdata1.layers['raw'].copy())
        sdata1.X = sdata1.layers['raw'].copy()
    else:
        if type(sdata1.X) != scipy.sparse._csr.csr_matrix:
            sdata1.X = csr(sdata1.X.copy())
        sdata1.layers['raw'] = sdata1.X.copy()
    if mode == 'default':
        sc.pp.normalize_total(sdata1)
        sdata1.layers['norm'] = csr(sdata1.X.copy())
        sc.pp.log1p(sdata1)
        sdata1.layers['log1p_norm'] = csr(sdata1.X.copy())
        sc.pp.scale(sdata1,zero_center = zero_center)
        if scipy.sparse.issparse(sdata1.X): #### automatically change to non csr matrix (zero_center == True, the .X would be sparce)
            sdata1.X = sdata1.X.toarray().copy()
        sdata1.layers['log1p_norm_scaled'] = sdata1.X.copy()
        if regressout:
            sdata1.obs['total_counts'] = sdata1.layers['raw'].toarray().sum(axis=1)
            sc.pp.regress_out(sdata1, ['total_counts'])
            sdata1.layers['log1p_norm_scaled'] = sdata1.X.copy()
        return sdata1 #### sdata1.X is sdata1.layers['log1p_norm_scaled']
    elif mode == 'customized':
        if target_sum == 1e4:
            target_sum_str = '1e4'
        else:
            target_sum_str = str(target_sum)
        sc.pp.normalize_total(sdata1,target_sum=target_sum)
        sdata1.layers[f'norm{target_sum_str}'] = csr(sdata1.X.copy())
        sc.pp.log1p(sdata1,base = base)
        sdata1.layers[f'log{str(base)}_norm{target_sum_str}'] = csr(sdata1.X.copy())
        sc.pp.scale(sdata1,zero_center = zero_center)
        if scipy.sparse.issparse(sdata1.X): #### automatically change to non csr matrix (zero_center == True, the .X would be sparce)
            sdata1.X = sdata1.X.toarray().copy()
        sdata1.layers[f'log{str(base)}_norm{target_sum_str}_scaled'] = sdata1.X.copy()
        if regressout:
            sdata1.obs['total_counts'] = sdata1.layers['raw'].toarray().sum(axis=1)
            sc.pp.regress_out(sdata1, ['total_counts'])
            sdata1.layers[f'log{str(base)}_norm{target_sum_str}_scaled'] = sdata1.X.copy()
        return sdata1 #### sdata1.X is sdata1.layers[f'log{str(base)}_norm{target_sum_str}_scaled']
    else:
        print('Please set the `mode` as one of the {"default", "customized"}.')
