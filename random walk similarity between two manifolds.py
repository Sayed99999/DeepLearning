import numpy as np
import csv
import networkx as nx
from sklearn.datasets import make_swiss_roll
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import networkx as nx
from scipy.sparse import csr_matrix

#define random walk distance function.
def random_walk_distance(graph1, graph2, t):
    n1 = graph1.number_of_nodes()
    n2 = graph2.number_of_nodes()
    P1 = csr_matrix(nx.to_numpy_array(graph1))
    P2 = csr_matrix(nx.to_numpy_array(graph2))
    I1 = np.eye(n1)
    I2 = np.eye(n2)
    W1 = np.linalg.inv(I1 - t * P1.toarray())
    W2 = np.linalg.inv(I2 - t * P2.toarray())
    return np.linalg.norm(W1 - W2)

# define initial state of all varibles
n_samples = 2000
noise1 = 0.0
noise2 = 0.0
random_state1 = 42
random_state2 = 42
t = 0.1  # Time step used in random walk


# create a container table to save the results of all varibles as they change
table = []
header = ['Shape', 'n_samples', 'noise1', 'noise2', 'difference' ]

# Run the loop 100 times
for i in range(100):

    #create two manifold structures
    X1, y1 = make_swiss_roll(n_samples=n_samples, noise=noise1, random_state=random_state1)
    X2, y2 = make_swiss_roll(n_samples=n_samples, noise=noise2, random_state=random_state2)


    # Create graphs from the datasets- nodes are connected based on 3 nearest neighbours 
    k = 3  # Number of nearest neighbors
    graph1 = kneighbors_graph(X1, k, mode='distance')
    graph2 = kneighbors_graph(X2, k, mode='distance')

    # Convert graphs to NetworkX graph objects
    graph1_nx = nx.from_scipy_sparse_matrix(graph1)
    graph2_nx = nx.from_scipy_sparse_matrix(graph2)

    # Calculate Random Walk Distance (RWD)
    rwd = random_walk_distance(graph1_nx, graph2_nx, t)

    # increase the noise by 1
    #noise1 = 0.05
    noise2 = noise2 + 0.01

    # Store the values in a row of the table
    row = ['Swiss roll', n_samples, noise1, noise2, rwd]
    table.append(row)


# Save the table to a CSV file
with open('results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(table)
    

print("Results saved successfully.")







