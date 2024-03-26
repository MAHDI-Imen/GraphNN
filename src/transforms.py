import torch
from torch_geometric.transforms import BaseTransform
from utils import degree

class AddDegreeFeature(BaseTransform):
    """Adds the node degree as featue.
    """
    def __call__(self, data):
        deg = degree(data.edge_index).unsqueeze(-1).to(torch.float)
        if data.x is not None:
            data.x = torch.cat([data.x, deg], dim=-1)
        else:
            data.x = deg
        return data