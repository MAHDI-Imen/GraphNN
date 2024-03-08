import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import degree, add_self_loops


class MessagePassingLayer(nn.Module):
    """
    Applies a message passing layer to the graph.

    Args:
        aggr: The aggregation function to use. This function should take in a tensor of shape (num_nodes, num_features) and return a tensor of shape (num_nodes, num_features).
        msg: The message function to use. This function should take in a tensor of shape (num_nodes, num_features) and a tensor of shape (num_edges, 2) and return a tensor of shape (num_edges, num_features).
        update: The update function to use. This function should take in a tensor of shape (num_nodes, num_features) and a tensor of shape (num_nodes, num_features) and return a tensor of shape (num_nodes, num_features).
    """
    def __init__(self, aggr, msg, update):
        self.aggr = self._get_aggr_function(aggr)
        self.msg = msg
        self.update = update

    def propagate(self, x, edge_index=None):
        return self.update(x, self.aggr(self.msg(x, edge_index), edge_index))
    
    def _get_msg_function(self, msg):
        if msg is None:
            return lambda x, edge_index: x
        else:
            return msg
    
    def _get_update_function(self, update):
        if update is None:
            return lambda x, x_agg: x_agg
        else:
            return update
    
    def _get_aggr_function(self, aggr):
        if type(aggr) == str:
            if aggr == 'mean':
                return lambda x, edge_index: self._mean_aggr(x, edge_index)
            elif aggr == 'sum':
                return lambda x, edge_index: self._add_aggr(x, edge_index)
            elif aggr == 'max':
                pass
        else:
            return aggr

    def _add_aggr(self, x, edge_index=None):
        if edge_index is None:
            return x
        index = edge_index[0].repeat(x.shape[1], 1).T
        src = x[edge_index[1]]
        aggregated = torch.zeros_like(x).scatter_add_(0, index=index, src=src)
        return aggregated
    
    def _mean_aggr(self, x, edge_index=None):
        if edge_index is None:
            return x
        aggregated = self._add_aggr(x, edge_index)
        degrees = degree(edge_index, num_nodes=x.shape[0])
        # if x has more than one feature, we need to divide each feature by the degree
        if len(x.shape) > 1:
            degrees = degrees.view(-1, 1).repeat(1, x.shape[1])
        # replace 0s with 1s to avoid division by zero
        degrees[degrees == 0] = 1
        return aggregated / degrees
    
    def _max_aggr(self, x, edge_index=None):
        raise NotImplementedError


class GCNConvLayer(MessagePassingLayer):
    """
    Applies a Graph Convolutional Network (GCN) layer to the graph.

    Args:
        in_features: The number of input features.
        out_features: The number of output features.
        aggr: The aggregation function to use. This function should take in a tensor of shape (num_nodes, num_features) and return a tensor of shape (num_nodes, num_features).
    """
    def __init__(self, in_features, out_features, aggr='mean'):
        self.in_features = in_features
        self.out_features = out_features
        self.aggr = aggr
        self.msg = self._get_msg_function()
        self.update = self._get_update_function()
        self.linear = nn.Linear(in_features, out_features)
    
    def _get_msg_function(self):
        return lambda x, edge_index: x
    
    def _get_update_function(self):
        return lambda x, x_aggr: F.relu(self.linear(x_aggr))
    
    def forward(self, x, edge_index):
        return self.propagate(x, edge_index)