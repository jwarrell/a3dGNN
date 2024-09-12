import numpy as np
from collections import defaultdict
import time
from sklearn.cluster import AffinityPropagation
import torch
from sklearn.cluster import DBSCAN

def cluster_label2list(label):
    cluster=[[] for _ in range(np.max(label)+1)] 
    for i in range(len(label)):
        cluster[label[i]].append(i)
    return cluster

def calc_degree(graph_dict,index):
    return len(graph_dict[index])

def graph_dictionary(edge_first,edge_second,n_nodes):
    graph_dict = defaultdict(list)

    for i in range(len(edge_first)):
        a, b = edge_first[i], edge_second[i]    
        # Creating the graph as adjacency list
        graph_dict[a].append(b)
        graph_dict[b].append(a)
    if len(graph_dict)!=n_nodes:
        for index in range(n_nodes):
            graph_dict[index]

    return graph_dict

def GWJV_SP(graph_dict):
    max_level=5 #threshold for max distance
    n_nodes=len(graph_dict)
    SP=np.zeros((n_nodes,n_nodes))-1
    for node in range(n_nodes):
        n_level=0
        current_level = [node]
        current_distance = 0
        SP[node][node] = 0
        while len(current_level)>0 and current_distance<max_level:
            current_distance += 1
            next_level = []
            for n in current_level:
                for ne in graph_dict[n]:
                    if SP[node][ne] == -1:
                        next_level.append(ne)
                        SP[node][ne] = current_distance
            current_level = next_level
            n_level+=1
    SP[SP == -1] = max_level
    return SP

def euclidean_distance(a, b):
    return torch.norm(a - b, dim=-1)
def remove_edges_by_distance_and_type(x, pos, old_edges, net_mhc, net_peptide, threshold):
    # breakpoint()
    feature = torch.cat([pos, x], dim=-1)
    feature_mhc = net_mhc.forward(feature[:, :-1])
    feature_peptide = net_peptide.forward(feature[:, :-1])
    # Separate the edge tensor E into source and target node indices
    source_indices = old_edges[0]
    target_indices = old_edges[1]

    # Get the feature vectors for the source and target nodes
    source_features_mhc = feature_mhc[source_indices]
    target_features_mhc = feature_mhc[target_indices]

    source_features_peptide = feature_peptide[source_indices]
    target_features_peptide = feature_peptide[target_indices]

    # Get the node types for the source and target nodes
    source_types = torch.round(feature[source_indices, -1])
    target_types = torch.round(feature[target_indices, -1])

    source_features_mhc[source_types == 1] = source_features_peptide[source_types == 1]
    target_features_mhc[target_types == 1] = target_features_peptide[target_types == 1]
    source_features = source_features_mhc
    target_features = target_features_mhc

    # Compute the Euclidean distances between feature vectors
    distances = euclidean_distance(source_features, target_features)

    # Create a mask for edges that should be removed based on distance and type
    mask = (distances <= threshold) | (source_types == target_types)

    # Apply the mask to keep only valid edges
    valid_source_indices = source_indices[mask]
    valid_target_indices = target_indices[mask]
    valid_edges = torch.stack([valid_source_indices, valid_target_indices], dim=0)

    return valid_edges

def remove_edges_by_distance(x, pos, old_edges, net_mhc, net_peptide, threshold):
    # breakpoint()
    feature = torch.cat([pos, x], dim=-1)
    feature_mhc = net_mhc.forward(feature[:, :-1])
    feature_peptide = net_peptide.forward(feature[:, :-1])
    # Separate the edge tensor E into source and target node indices
    source_indices = old_edges[0]
    target_indices = old_edges[1]

    # Get the feature vectors for the source and target nodes
    source_features_mhc = feature_mhc[source_indices]
    target_features_mhc = feature_mhc[target_indices]

    source_features_peptide = feature_peptide[source_indices]
    target_features_peptide = feature_peptide[target_indices]

    # Get the node types for the source and target nodes
    source_types = torch.round(feature[source_indices, -1])
    target_types = torch.round(feature[target_indices, -1])

    source_features_mhc[source_types == 1] = source_features_peptide[source_types == 1]
    target_features_mhc[target_types == 1] = target_features_peptide[target_types == 1]
    source_features = source_features_mhc
    target_features = target_features_mhc

    # Compute the Euclidean distances between feature vectors
    distances = euclidean_distance(source_features, target_features)

    # Create a mask for edges that should be removed based on distance and type
    mask = distances <= threshold

    # Apply the mask to keep only valid edges
    valid_source_indices = source_indices[mask]
    valid_target_indices = target_indices[mask]
    valid_edges = torch.stack([valid_source_indices, valid_target_indices], dim=0)

    return valid_edges

def add_new_edges(E, A, threshold, item_type):
    # Get the indices of nodes that satisfy the distance threshold
    # breakpoint()
    row_indices, col_indices = torch.where(A < threshold)
    row_indices_cat = torch.cat([row_indices, col_indices])
    col_indices_cat = torch.cat([col_indices, row_indices])
    valid_indices = item_type[row_indices_cat] != item_type[col_indices_cat]

    new_edges = torch.stack([row_indices_cat, col_indices_cat], dim=0)[:, valid_indices]

    # Concatenate new edges with the original edge tensor E
    updated_E = torch.cat([E, new_edges], dim=1)

    # Remove duplicate edges
    updated_E, _ = torch.unique(updated_E, sorted=False, return_inverse=True, dim=1)

    return updated_E
def add_edges_by_distance_and_type(x, pos, old_edges, net_mhc, net_peptide, threshold):
    # breakpoint()
    # breakpoint()
    feature = torch.cat([pos, x], dim=-1)
    feature_mhc = net_mhc.forward(feature[:, :-1])
    feature_peptide = net_peptide.forward(feature[:, :-1])
    # Separate the edge tensor E into source and target node indices

    item_type = feature[:, -1]
    feature_mhc[item_type == 1] = feature_peptide[item_type == 1]
    feature = feature_mhc
    distances = torch.cdist(feature, feature)

    new_edges = add_new_edges(old_edges, distances, threshold, item_type)

    return new_edges

# def add_new_edges(E, A, threshold):
#     # Get the indices of nodes that satisfy the distance threshold
#     row_indices, col_indices = torch.where(A < threshold)
#     row_indices = torch.cat([row_indices, col_indices])
#     col_indices = torch.cat([col_indices, row_indices])
#
#     # Create a mask to filter out existing edges in E
#     existing_edges_mask = ~torch.any(E[:, :, None] == torch.stack([row_indices, col_indices]), dim=0).all(dim=0)
#
#     # Filter the new edges based on the mask
#     new_row_indices = row_indices[existing_edges_mask]
#     new_col_indices = col_indices[existing_edges_mask]
#
#     # Concatenate the new edges to the original tensor E
#     new_edges = torch.stack([new_row_indices, new_col_indices], dim=0)
#     updated_E = torch.cat([E, new_edges], dim=1)
#
#     return updated_E
#
# def removing_edges(x, pos, net_mhc, net_peptide, old_edges, threshold):
#     mhc_index = x[:, -1] == 0
#     peptide_index = x[:, -1] == 1
#     features = torch.cat([x, pos], dim=-1)
#     features_mhc = features[mhc_index]
#     features_peptide = features[peptide_index]
#     out_mhc = net_mhc(features_mhc)
#     out_peptide = net_peptide(features_peptide)
#     dist = torch.cdist(out_mhc, out_peptide)
#     new_edges = add_new_edges(old_edges, dist, threshold)
#     return new_edges

def AffinityClustering(x,edges,degree_matrix,distance_matrix,current_batch,physical_params):
    p_factor=physical_params[0].item()
    batch_offset=0
    nodes=x.detach().numpy()
    max_cluster=0
    cluster_label=[]
    for index,my_batch in enumerate(current_batch):
        my_n_nodes=len(my_batch.x)
        my_x=nodes[batch_offset:batch_offset+my_n_nodes]
        
        C_sp=physical_params[1].item()
        C_degree=physical_params[2].item()
        #print('cluster_params: ',p_factor,C_sp,C_degree)
        dis=my_x[:,None]-my_x[None,:]
        feature=np.linalg.norm(dis,axis=2)
        S=-(C_degree*degree_matrix[index]+C_sp*distance_matrix[index]+feature)
        S_median=np.median(S.flatten())
        np.fill_diagonal(S,S_median)
        preference=S_median*p_factor
        af=AffinityPropagation(damping=0.88,max_iter=1600,preference=preference,affinity='precomputed',random_state=0)
        af_label = af.fit_predict(S)
        af_label=af_label+max_cluster
        #print('cluster_size',len(feature)/max(af_label))
        cluster_label=np.append(cluster_label,af_label)
        max_cluster=max_cluster+max(af_label)+1
        batch_offset=batch_offset+my_n_nodes
        # damping 0.88 and max_iter=1600 help to remove all convergence issues
        # current cluster size \approx 30 for S_median*2, 80 for S_median*10, 40 for S_median*5

    edges=edges.numpy().T
    to_remove=[]

    for index,edge in enumerate(edges):
        if cluster_label[edge[0]]!=cluster_label[edge[1]]:
            to_remove.append(index)
    edges=np.delete(edges,obj=to_remove, axis=0)
    t = torch.from_numpy(edges.T)

    return t

def predetermined_cluster(edges,cluster_final):
    edges=edges.numpy().T
    to_remove=[]

    for index,edge in enumerate(edges):
        if cluster_final[edge[0]]!=cluster_final[edge[1]]:
            to_remove.append(index)

    edges=np.delete(edges,obj=to_remove, axis=0)
    t = torch.from_numpy(edges.T)

    return t

def AffinityClustering_oneGraph(x,edges,degree_matrix,distance_matrix,physical_params):
    p_factor=physical_params[0]

    C_sp=physical_params[1]
    C_degree=physical_params[2]

    dis=x[:,None]-x[None,:]
    feature=np.linalg.norm(dis,axis=2)
    S=-(C_degree*degree_matrix+C_sp*distance_matrix+feature)
    S_median=np.median(S.flatten())
    np.fill_diagonal(S,S_median)
    preference=S_median*p_factor
    af=AffinityPropagation(damping=0.88,max_iter=1600,preference=preference,affinity='precomputed',random_state=0)
    af_label = af.fit_predict(S)


    edges=edges.numpy().T
    to_remove=[]

    for index,edge in enumerate(edges):
        if af_label[edge[0]]!=af_label[edge[1]]:
            to_remove.append(index)
    edges=np.delete(edges,obj=to_remove, axis=0)
    t = torch.from_numpy(edges.T)

    return t


def calc_distance_matrix(graph_dict):
    SP=GWJV_SP(graph_dict)
    return SP

def calc_degree_matrix(graph_dict):
    n_nodes=len(graph_dict)
    degree_matrix=np.zeros((n_nodes,n_nodes))
    
    for i in range(n_nodes):
        degree_i=calc_degree(graph_dict,i)
        for j in range(i+1,n_nodes):
            degree_j=calc_degree(graph_dict,j)
            degree_matrix[i][j]=abs(degree_i-degree_j)
            degree_matrix[j][i]=degree_matrix[i][j]
    return degree_matrix

def calc_euclidean_distance_matrix(pos):
    # pos = torch.stack([i.pos for i in x], dim=0)
    pos = torch.stack([torch.from_numpy(i) for i in pos], dim=0)
    return torch.cdist(pos, pos)
