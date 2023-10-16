from os.path import exists

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch_geometric.utils import to_dense_adj


def compute(dl, filename):
    df = pd.DataFrame(columns=['num_nodes', 'num_edges', 'density', 'cluster_coefficient', 'centralization'])
    for _, batch in enumerate(dl):
        num_nodes = batch.shape[2]
        graph = batch[0][0]
        num_edges = count_edges(graph)
        density = compute_density(graph)
        cluster_coefficient = compute_cluster_coefficient(graph)
        centralization = compute_centralization(graph)

        df = pd.concat([df, pd.DataFrame({'num_nodes': num_nodes, 'num_edges': num_edges, 'density': density,
                        'cluster_coefficient': cluster_coefficient, 'centralization': centralization}, index=[0])], ignore_index=True)

    df = df.groupby('num_nodes').mean()
    df.to_csv(filename)


def count_edges(graph):
    return torch.sum(graph[graph == 1])


def compute_density(graph):
    num_nodes = graph.shape[0]
    num_edges = torch.sum(graph[graph == 1])
    num_possible_edges = num_nodes * (num_nodes - 1)
    return num_edges / num_possible_edges


def compute_cluster_coefficient(graph):
    adj_sq = torch.matmul(graph, graph)
    triangles = torch.diag(adj_sq).float()
    degrees = graph.sum(dim=1).float()
    cluster_coeffs = triangles / (degrees * (degrees - 1))
    cluster_coeffs[torch.isnan(cluster_coeffs)] = 0
    cluster_coeffs[torch.isinf(cluster_coeffs)] = 0
    return cluster_coeffs.mean()


def compute_centralization(graph):
    degrees = graph.sum(dim=1).float()
    max_deg = degrees.max()
    centr = (max_deg - degrees).sum()
    n = graph.shape[0]
    norm_factor = 1 if n <= 2 else 1 / ((n - 1) * (n - 2))
    return norm_factor * centr


def main():
    df = pd.DataFrame(columns=['num_nodes', 'num_edges', 'density', 'cluster_coefficient', 'centralization'])
    for num_nodes in [128, 256, 512, 1024]:
        for i in range(10):
            if exists(f"output/graphs/cpg_{num_nodes}_{i}.pt"):
                graph = torch.load(f"output/graphs/cpg_{num_nodes}_{i}.pt", map_location=torch.device('cpu'))
                adj = to_dense_adj(graph.edge_index)[0]

                num_edges = count_edges(adj)
                density = compute_density(adj)
                cluster_coefficient = compute_cluster_coefficient(adj)
                centralization = compute_centralization(adj)

                df = pd.concat([df, pd.DataFrame({'num_nodes': num_nodes, 'num_edges': num_edges, 'density': density,
                                                  'cluster_coefficient': cluster_coefficient,
                                                  'centralization': centralization}, index=[0])], ignore_index=True)

    df = df.groupby('num_nodes').mean()
    df.to_csv("stats_generated.csv")


if __name__ == "__main__":
    main()
