import torch
import torchvision
import torchvision.transforms as transforms

from torch_geometric.data import Data

def load_mnist_graph(subset=None):
    # load data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    if subset is not None:
        trainset = torch.utils.data.Subset(trainset, range(subset))
        testset = torch.utils.data.Subset(testset, range(subset))
    # define edge index for an image
    coordinates_to_index = lambda i, j: int(i*28+j)
    edge_index = []
    for i in range(28):
        for j in range(28):
            if i+1<28:
                edge_index.append([coordinates_to_index(i, j), coordinates_to_index(i+1, j)])
            if j+1<28:
                edge_index.append([coordinates_to_index(i, j), coordinates_to_index(i, j+1)])
            if i-1>=0:
                edge_index.append([coordinates_to_index(i, j), coordinates_to_index(i-1, j)])
            if j-1>=0:
                edge_index.append([coordinates_to_index(i, j), coordinates_to_index(i, j-1)])
            if i+1<28 and j+1<28:
                edge_index.append([coordinates_to_index(i, j), coordinates_to_index(i+1, j+1)])
            if i+1<28 and j-1>=0:
                edge_index.append([coordinates_to_index(i, j), coordinates_to_index(i+1, j-1)])
            if i-1>=0 and j+1<28:
                edge_index.append([coordinates_to_index(i, j), coordinates_to_index(i-1, j+1)])
            if i-1>=0 and j-1>=0:
                edge_index.append([coordinates_to_index(i, j), coordinates_to_index(i-1, j-1)])
    edge_index = torch.tensor(edge_index, dtype=torch.int64).t().contiguous()

    train_graphs = [Data(x=x.view(28*28,1), edge_index=edge_index, y=y) for x, y in trainset]
    test_graphs = [Data(x=x.view(28*28,1), edge_index=edge_index, y=y) for x, y in testset]

    return train_graphs, test_graphs

