import networkx as nx

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
