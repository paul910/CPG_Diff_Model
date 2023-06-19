import pandas as pd
import torch

from config import Config
from dataloader import get_adj_dataloader


def compute_edges_per_graph(dataloader):
    stats = {}
    for _, batch in enumerate(dataloader):
        num_nodes = batch.shape[2]
        graph = batch[0][0]
        num_edges = torch.sum(graph[graph == 1])

        if num_nodes not in stats.keys():
            stats[num_nodes] = [num_edges]
        else:
            stats[num_nodes].append(num_edges)

    out = {}
    for key in stats.keys():
        out[key] = float(sum(stats[key]) / len(stats[key]))

    return out


def compute_density(dataloader):
    stats = {}
    for _, batch in enumerate(dataloader):
        num_nodes = batch.shape[2]
        graph = batch[0][0]
        num_edges = torch.sum(graph[graph == 1])
        num_possible_edges = num_nodes * (num_nodes - 1)
        density = num_edges / num_possible_edges

        if num_nodes not in stats.keys():
            stats[num_nodes] = [density]
        else:
            stats[num_nodes].append(density)

    out = {}
    for key in stats.keys():
        out[key] = float(sum(stats[key]) / len(stats[key]))

    return out


def compute_cluster_coefficient(dataloader):
    stats = {}
    # Compute the cluster coefficient of the graph
    for _, batch in enumerate(dataloader):
        num_nodes = batch.shape[2]
        graph = batch[0][0]
        adj_sq = torch.matmul(graph, graph)
        triangles = torch.diag(adj_sq).float()
        degrees = graph.sum(dim=1).float()
        cluster_coeffs = triangles / (degrees * (degrees - 1))
        cluster_coeffs[torch.isnan(cluster_coeffs)] = 0
        cluster_coeffs[torch.isinf(cluster_coeffs)] = 0
        cluster_coeff = cluster_coeffs.mean()

        if num_nodes not in stats.keys():
            stats[num_nodes] = [cluster_coeff]
        else:
            stats[num_nodes].append(cluster_coeff)

    out = {}
    for key in stats.keys():
        out[key] = float(sum(stats[key]) / len(stats[key]))

    return out


def compute_centralization(dataloader):
    # Compute the centralization of the graph
    stats = {}

    for _, batch in enumerate(dataloader):
        num_nodes = batch.shape[2]
        graph = batch[0][0]
        degrees = graph.sum(dim=1).float()
        max_deg = degrees.max()
        centralization = (max_deg - degrees).sum()
        n = graph.shape[0]
        norm_factor = 1 if n <= 2 else 1 / ((n - 1) * (n - 2))
        centralization = norm_factor * centralization

        if num_nodes not in stats.keys():
            stats[num_nodes] = [centralization]
        else:
            stats[num_nodes].append(centralization)

    out = {}
    for key in stats.keys():
        out[key] = float(sum(stats[key]) / len(stats[key]))

    return out


def compute_graph_statistics():
    dataloader = get_adj_dataloader(Config.DATA_PATH, Config.BATCH_SIZE, Config.MODEL_DEPTH)
    edges_per_graph = compute_edges_per_graph(dataloader)
    density = compute_density(dataloader)
    cluster_coefficient = compute_cluster_coefficient(dataloader)
    centralization = compute_centralization(dataloader)

    edges_per_graph = pd.DataFrame.from_dict(edges_per_graph, orient='index', columns=['Avg. Edges'])
    density = pd.DataFrame.from_dict(density, orient='index', columns=['Avg. Density'])
    cluster_coefficient = pd.DataFrame.from_dict(cluster_coefficient, orient='index', columns=['Avg. Cluster Coefficient'])
    centralization = pd.DataFrame.from_dict(centralization, orient='index', columns=['Avg. Centralization'])

    combined = pd.concat([edges_per_graph, density, cluster_coefficient, centralization], axis=1).sort_index()
    combined.to_csv("stats/statistics.csv")


if __name__ == "__main__":
    compute_graph_statistics()
