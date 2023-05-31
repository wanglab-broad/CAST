import torch, dgl
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
# from libpysal import weights
# from libpysal.cg import voronoi_frames
# import seaborn as sns
# from sklearn.cluster import KMeans

from .models.aug import random_aug
from .utils import coords2adjacentmat
from .visualize import kmeans_plot_multiple, add_scale_bar
from timeit import default_timer as timer
from collections import OrderedDict

def train_seq(graphs, args, dump_epoch_list, out_prefix, model):
    """The CAST MARK training function

    Args:
        graphs (List[Tuple(str, dgl.Graph, torch.Tensor)]): List of 3-member tuples, each tuple represents one tissue sample, containing sample name, a DGL graph object, and a feature matrix in the torch.Tensor format
        args (model_GCNII.Args): the Args object contains training parameters
        dump_epoch_list (List): A list of epoch id you hope training snapshots to be dumped, for debug use, empty by default
        out_prefix (str): file name prefix for the snapshot files
        model (model_GCNII.CCA_SSG): the GNN model

    Returns:
        Tuple(Dict, List, CCA_SSG): returns a 3-member tuple, a dictionary containing the graph embeddings for each sample, a list of every loss value, and the trained model object
    """    
    model = model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr1, weight_decay=args.wd1)

    loss_log = []
    time_now = timer()

    for epoch in range(args.epochs):

        with torch.no_grad():
            if epoch in dump_epoch_list:
                model.eval()
                dump_embedding = OrderedDict()
                for name, graph, feat in graphs:
                    # graph = graph.to(args.device)
                    # feat = feat.to(args.device)
                    dump_embedding[name] = model.get_embedding(graph, feat)
                torch.save(dump_embedding, f'{out_prefix}_embed_dict_epoch{epoch}.pt')
                torch.save(loss_log, f'{out_prefix}_loss_log_epoch{epoch}.pt')
                print(f"Successfully dumped epoch {epoch}")

        losses = dict()
        model.train()
        optimizer.zero_grad()
        print(f'Epoch: {epoch}')

                
        # for i in range(len(graphs)):
            # print(name)
        for name_, graph_, feat_ in graphs:
            # print(name_)
            with torch.no_grad():
                N = graph_.number_of_nodes()
                graph1, feat1 = random_aug(graph_, feat_, args.dfr, args.der)
                graph2, feat2 = random_aug(graph_, feat_, args.dfr, args.der)

                graph1 = graph1.add_self_loop()
                graph2 = graph2.add_self_loop()

            z1, z2 = model(graph1, feat1, graph2, feat2)

            c = torch.mm(z1.T, z2)
            c1 = torch.mm(z1.T, z1)
            c2 = torch.mm(z2.T, z2)

            c = c / N
            c1 = c1 / N
            c2 = c2 / N

            loss_inv = - torch.diagonal(c).sum()
            iden = torch.eye(c.size(0), device=args.device)
            loss_dec1 = (iden - c1).pow(2).sum()
            loss_dec2 = (iden - c2).pow(2).sum()
            loss = loss_inv + args.lambd * (loss_dec1 + loss_dec2)
            loss.backward()
            optimizer.step()
            
        # del graph1, feat1, graph2, feat2        
        loss_log.append(loss.item())
        time_step = timer() - time_now
        time_now += time_step
        print(f'Loss: {loss.item()} step time={time_step:.3f}s')
    
    model.eval()
    with torch.no_grad():
        dump_embedding = OrderedDict()
        for name, graph, feat in graphs:
            dump_embedding[name] = model.get_embedding(graph, feat)
    return dump_embedding, loss_log, model

# graph construction tools
def delaunay_dgl(sample_name, df, output_path,if_plot=True):
    coords = np.column_stack((np.array(df)[:,0],np.array(df)[:,1]))
    delaunay_graph = coords2adjacentmat(coords,output_mode = 'raw')
    if if_plot:
        positions = dict(zip(delaunay_graph.nodes, coords))
        _, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        nx.draw(
            delaunay_graph,
            positions,
            ax=ax,
            node_size=1,
            node_color="#000000",
            edge_color="#5A98AF",
            alpha=0.6,
        )
        plt.axis('equal')
        plt.savefig(f'{output_path}/delaunay_{sample_name}.png')
    return dgl.from_networkx(delaunay_graph)

#################### Visualization ####################

# def kmeans_plot_multiple(embed_dict_t,graph_list,coords,taskname_t,output_path_t,k=20,dot_size = 10,scale_bar_t = None):
#     num_plot = len(embed_dict_t)
#     plot_row = int(np.floor(num_plot/2) + 1)
#     embed_stack = embed_dict_t[graph_list[0]].cpu().detach().numpy()
#     for i in range(1,num_plot):
#         embed_stack = np.row_stack((embed_stack,embed_dict_t[graph_list[i]].cpu().detach().numpy()))
#     kmeans = KMeans(n_clusters=k,random_state=0).fit(embed_stack)
#     cell_label = kmeans.labels_
#     cluster_pl = sns.color_palette('tab20',len(np.unique(cell_label)))
#     plt.figure(figsize=((20,10 * plot_row)))
#     cell_label_idx = 0
#     for j in range(num_plot):
#         plt.subplot(plot_row,2,j+1)
#         coords0 = coords[graph_list[j]]
#         col=coords0[:,0].tolist()
#         row=coords0[:,1].tolist()
#         cell_type_t = cell_label[cell_label_idx:(cell_label_idx + coords0.shape[0])]
#         cell_label_idx += coords0.shape[0]
#         for i in set(cell_type_t):
#             plt.scatter(np.array(col)[cell_type_t == i],
#             np.array(row)[cell_type_t == i], s=dot_size,edgecolors='none',
#             c=np.array(cluster_pl)[cell_type_t[cell_type_t == i]],label = str(i), rasterized=True)
#         plt.title(graph_list[j] + ' (KMeans, k = ' + str(k) + ')',fontsize=20)
#         plt.xticks(fontsize=20)
#         plt.yticks(fontsize=20)
#         plt.axis('equal')
#         if (type(scale_bar_t) != type(None)):
#             add_scale_bar(scale_bar_t[0],scale_bar_t[1])
#     plt.savefig(f'{output_path_t}/{taskname_t}_trained_k{str(k)}.pdf',dpi = 300)
#     return cell_label

# def add_scale_bar(length_t,label_t):
#     import matplotlib.font_manager as fm
#     from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
#     fontprops = fm.FontProperties(size=20, family='Arial')
#     bar = AnchoredSizeBar(plt.gca().transData, length_t, label_t, 4, pad=0.1,
#                         sep=5, borderpad=0.5, frameon=False,
#                         size_vertical=0.1, color='black',fontproperties = fontprops)
#     plt.gca().add_artist(bar)