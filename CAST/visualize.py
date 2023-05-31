import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

def kmeans_plot_multiple(embed_dict_t,graph_list,coords,taskname_t,output_path_t,k=20,dot_size = 10,scale_bar_t = None,minibatch = False):
    num_plot = len(embed_dict_t)
    plot_row = int(np.floor(num_plot/2) + 1)
    embed_stack = embed_dict_t[graph_list[0]].cpu().detach().numpy()
    for i in range(1,num_plot):
        embed_stack = np.row_stack((embed_stack,embed_dict_t[graph_list[i]].cpu().detach().numpy()))
    print(f'Perform KMeans clustering on {embed_stack.shape[0]} cells...')
    kmeans = KMeans(n_clusters=k,random_state=0).fit(embed_stack) if minibatch == False else MiniBatchKMeans(n_clusters=k,random_state=0).fit(embed_stack)
    cell_label = kmeans.labels_
    cluster_pl = sns.color_palette('tab20',len(np.unique(cell_label)))
    print(f'Plotting the KMeans clustering results...')
    plt.figure(figsize=((20,10 * plot_row)))
    cell_label_idx = 0
    for j in range(num_plot):
        plt.subplot(plot_row,2,j+1)
        coords0 = coords[graph_list[j]]
        col=coords0[:,0].tolist()
        row=coords0[:,1].tolist()
        cell_type_t = cell_label[cell_label_idx:(cell_label_idx + coords0.shape[0])]
        cell_label_idx += coords0.shape[0]
        for i in set(cell_type_t):
            plt.scatter(np.array(col)[cell_type_t == i],
            np.array(row)[cell_type_t == i], s=dot_size,edgecolors='none',
            c=np.array(cluster_pl)[cell_type_t[cell_type_t == i]],label = str(i), rasterized=True)
        plt.title(graph_list[j] + ' (KMeans, k = ' + str(k) + ')',fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.axis('equal')
        if (type(scale_bar_t) != type(None)):
            add_scale_bar(scale_bar_t[0],scale_bar_t[1])
    plt.savefig(f'{output_path_t}/{taskname_t}_trained_k{str(k)}.pdf',dpi = 100)
    return cell_label

def add_scale_bar(length_t,label_t):
    import matplotlib.font_manager as fm
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    fontprops = fm.FontProperties(size=20, family='Arial')
    bar = AnchoredSizeBar(plt.gca().transData, length_t, label_t, 4, pad=0.1,
                        sep=5, borderpad=0.5, frameon=False,
                        size_vertical=0.1, color='black',fontproperties = fontprops)
    plt.gca().add_artist(bar)