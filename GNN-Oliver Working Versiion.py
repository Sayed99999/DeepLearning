import torch
import torch.nn.functional as F
import numpy as np
import time
from sklearn.model_selection import train_test_split
import urllib.request

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt'
filename = 'data_banknote_authentication.txt'
urllib.request.urlretrieve(url, filename)

class DataSet():
    def __init__(self):
        self.x = None
        self.y = None
        self.train_mask = None
        self.val_mask = None
        self.test_mask = None
        self.edge_index = None
        self.edge_attr = None

        """
        x: This is a matrix that represents the node features for each node in the graph.
        y: This is a matrix that represents the target labels for each node in the graph.
        train_mask: This is a boolean vector that indicates which nodes in the graph are used for training.
        val_mask: This is a boolean vector that indicates which nodes in the graph are used for validation.
        test_mask: This is a boolean vector that indicates which nodes in the graph are used for testing.
        edge_index: This is a matrix that represents the edges in the graph. Each column of the matrix represents 
        an edge, and the values in the column represent the indices of the nodes that are connected by that edge.
        edge_attr: This is a matrix that represents the edge features for each edge in the graph.
        In the constructor (__init__ method), all these attributes are initialized to None. 
        The actual values for these attributes will be set later, depending on the specific dataset being used.
        """
    def preprocess(self, data, n_train):
        x = data[:, :-1]
        y = data[:, -1].astype(int)
        x = (x - x.mean(axis=0)) / x.std(axis=0)
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=n_train, stratify=y)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=n_train//10, stratify=y_train)
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.train_mask = torch.tensor(np.isin(y, y_train), dtype=torch.bool)
        self.val_mask = torch.tensor(np.isin(y, y_val), dtype=torch.bool)
        self.test_mask = torch.tensor(np.isin(y, y_test), dtype=torch.bool)
        
        """
        The function takes two arguments:
        
        data: a numpy array containing the features and labels for each data point in the dataset.
        n_train: an integer specifying the number of data points to use for training.
        
        The function performs the following steps:
        
        Separate the features (x) and labels (y) from the input data.
        Normalize the features by subtracting the mean and dividing by the standard deviation.
        Split the dataset into three subsets: training, validation, and testing sets. 
        The training set size is specified by the n_train argument, and the validation set size is set to be 
        one-tenth of the training set size.
        Convert the numpy arrays into PyTorch tensors and store them in the corresponding attributes of the DataSet 
        instance: self.x, self.y, self.train_mask, self.val_mask, and self.test_mask. The self.train_mask, self.val_mask, 
        and self.test_mask attributes are boolean tensors that indicate which data points belong to each respective set.
        
        Overall, this function preprocesses the input data and stores it in the DataSet object in a format that is 
        suitable for use in machine learning models that operate on graphs.
        
        """
    

    def adj_to_edge(self, adj):
        """
        

        Parameters
        ----------
        adj : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        N = adj.shape[0]
        edge_index = []
        edge_weight = []
        for i in range(N):
            for j in range(N):
                if adj[i, j] != 0:
                    edge_index.append([i, j])
                    edge_weight.append(adj[i, j])
        self.edge_index = torch.tensor(edge_index).T
        self.edge_weight = torch.tensor(edge_weight)


        """
        The function takes a single argument:
        
        adj: a numpy array representing the adjacency matrix of the graph.
        
        The function performs the following steps:
        
        Get the number of nodes in the graph (which is the size of the adjacency matrix).
        Initialize two empty lists, edge_index and edge_weight, to store the edges and edge weights.
        Iterate over all pairs of nodes in the graph. If the weight of the edge between node i and node j is not zero, 
        append the pair (i, j) to edge_index and append the weight of the edge to edge_weight.
        Convert the edge_index list to a PyTorch tensor and transpose it (so that the first row represents the 
        source nodes and the second row represents the destination nodes), and store it in the self.edge_index 
        attribute of the DataSet instance.
        
        Convert the edge_weight list to a PyTorch tensor and store it in the self.edge_weight attribute of the DataSet instance.
        
        Overall, this function converts an adjacency matrix representation of a graph into an edge list representation that 
        is more suitable for use in graph-based machine learning tasks.
        
        """


class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels, 16, 1)
        self.conv2 = torch.nn.Conv1d(16, out_channels, 1)
        
        """
        This function is a class definition for a Graph Convolutional Network (GCN) module in PyTorch.
        
        The class takes two arguments:
        
        in_channels: an integer specifying the number of input channels 
        (i.e., the number of features for each node in the input graph).
        
        out_channels: an integer specifying the number of output channels 
        (i.e., the number of features for each node in the output graph).
        
        The class defines two convolutional layers (self.conv1 and self.conv2) that will be used to 
        transform the input graph data. Specifically, the first convolutional layer (self.conv1) applies a 1D convolution 
        to the input data, with 16 output channels and a kernel of size 1. The second convolutional layer (self.conv2) 
        applies another 1D convolution to the output of the first layer, with out_channels output channels and a kernel size of 1.
        """
        

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x.permute(1, 0).unsqueeze(0)
        x = F.relu(self.conv1(x))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x)
        x = x.squeeze(0).permute(1, 0)
        return F.log_softmax(x, dim=1)


        """
        test without dropouts - will it help?
        
        This function is the forward() method of a PyTorch model class for a Graph Convolutional Network (GCN).
        
        The forward() method takes a single argument data, which is assumed to be an instance of a custom dataset class 
        that contains the graph data to be processed.
        
        The method first extracts the x and edge_index attributes from the data object. 
        x is a tensor containing the node features of the input graph, while edge_index is a tensor containing 
        the edge connections between nodes.
        
        Next, the method applies some transformations to the x tensor, Specifically, it permutes the dimensions of the tensor 
        to be compatible with the convolutional layers, unsqueezes the tensor to add a batch dimension
        (By adding a batch dimension, unsqueeze allows us to process multiple inputs in 
         parallel, which can significantly speed up the training process) 
        
        and applies the first convolutional layer (self.conv1) to the tensor.
        These operations are necessary to ensure that the input tensor has the correct shape expected by the 
        Conv1d layer, allowing the layer to correctly process the input data.
        
        
        The output of the first convolutional layer is then passed through a ReLU activation function max(0,x), followed 
        by a dropout layer to prevent overfitting. The result is then passed through the second convolutional 
        layer (self.conv2) to produce the final output tensor.
        
        
        Finally, the output tensor is transformed back to its original shape (i.e., with the batch dimension removed 
        and the dimensions permuted back to their original order), and a log softmax activation function is applied 
        to produce the final output probabilities for each class.
        
        
        Overall, this forward() method performs the forward pass of a GCN model on the input graph data, 
        transforming the node features and edge connections into a set of class probabilities for the graph nodes.
        """


def compute_adjacency_matrix(data, beta = 1.0, binary = None):
    adj = torch.exp(-beta * torch.cdist(data, data))
    
    if binary:
        if type(binary) != float and type(binary) != int:
            binary = 0.5
        adj[adj < binary] = 0
        adj[adj > 0] = 1
        
    return adj

        """
        This function compute_adjacency_matrix takes a data tensor and computes a distance matrix between all pairs of points 
        using Euclidean distance. The resulting distance matrix is used to create an adjacency matrix, 
        which represents the edges between nodes in the graph.
        
        The function accepts two optional arguments beta and binary. 
        The beta parameter determines the decay rate of the distance function used to compute the adjacency matrix. 
        The binary parameter specifies whether to convert the resulting adjacency matrix to a binary matrix or not.
        
        If the binary parameter is set to a non-zero value, the resulting adjacency matrix is thresholded to create 
        a binary matrix. Specifically, all values in the adjacency matrix that are below the threshold value are set to zero, 
        and all values above the threshold value are set to one. If binary is not provided or is set to zero, 
        the adjacency matrix is returned as-is.
        
        The function returns the resulting adjacency matrix as a tensor object.
        """
        
def select_few(data, num):
    new_train_mask = torch.zeros_like(data.train_mask)
    classes = torch.unique(data.y[data.train_mask])
    for c in classes:
        idx = torch.where(data.y == c)[0]
        perm = torch.randperm(idx.shape[0])
        new_train_mask[idx[perm[:num]]] = True
    return new_train_mask
        """
        the function first initializes a new binary tensor new_train_mask with the same shape as the train_mask 
        attribute of the input data. It then iterates over the unique classes in the training set and selects 
        num samples from each class by generating a random permutation of the indices corresponding to the class samples 
        and selecting the first num indices from the permutation. The resulting indices are used to set the corresponding 
        entries of new_train_mask to True. The function then returns the resulting binary tensor new_train_mask.
        """

d = np.loadtxt(filename, delimiter=',')
RUNS = 10
results = np.zeros((RUNS,1))

start = time.time()
for run in range(RUNS):

    data = DataSet()
    data.preprocess(d, n_train=400)

    # compute graph matrices
    adj = compute_adjacency_matrix(data.x[data.train_mask], beta=1.0, binary=0.1)
    data.adj_to_edge(adj)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create a new model, 4 features / node, 2 classes
    model = GCN(4, 2).to(device)
    # we keep 10 examples per class labelled
    new_train_mask = select_few(data, 10)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(500):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[new_train_mask], data.y[new_train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    pred = model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f'Run {run+1}/{RUNS} accuracy: {acc:.4f}')

    results[run, 0] = acc

end = time.time()
print(f'Time for {RUNS} runs: {end - start}')
print(f'Best result: {np.max(results):.4f}')
print(f'Mean result: {np.mean(results):.4f}')

"""
This code performs a graph classification task using a simple graph convolutional network (GCN) model.

Firstly, it loads data from a file using np.loadtxt function.
Then, for RUNS number of times, it pre-processes the data using the DataSet class and computes the adjacency matrix 
using the compute_adjacency_matrix function.
Next, it initializes a GCN model with 4 input features per node and 2 output classes, and trains it for 
500 epochs using the Adam optimizer and negative log-likelihood loss (F.nll_loss) on a subset of the training data 
specified by new_train_mask.

After training, it evaluates the model on the test set and records the accuracy.
Finally, it prints the average and best accuracy over all runs, as well as the total time taken.

"""
