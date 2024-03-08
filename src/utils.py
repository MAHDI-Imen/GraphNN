import torch

def degree(edge_index, num_nodes=None):
    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1
        
    return torch.bincount(edge_index[0], minlength=num_nodes)

def add_self_loops(edge_index, num_nodes=None):
    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1
    num_nodes = max(num_nodes, int(edge_index.max()) + 1)

    loop_index = torch.arange(0, num_nodes, dtype=edge_index.dtype, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index, loop_index], dim=1)

    return edge_index