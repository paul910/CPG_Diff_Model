import pandas as pd
import torch
from torch.utils.data import DataLoader


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
    num_nodes = 256
    out = None
    for i in range(9):
        adj = torch.load(f"output/graphs/graph_{num_nodes}_{i}.pt", map_location=torch.device('cpu'))
        adj = torch.from_numpy(adj).unsqueeze(0).unsqueeze(0)

        if out is None:
            out = adj
        else:
            out = torch.cat((out, adj), 0)

    dataloader = DataLoader(out, batch_size=1, shuffle=True)

    compute(dataloader, f"stats/graph_{num_nodes}_stats.csv")


if __name__ == "__main__":
    main()
