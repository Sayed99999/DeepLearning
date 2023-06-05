import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from sklearn.datasets import make_swiss_roll
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
# Define random walk distance function
def random_walk_distance(adj1, adj2, t):
    n1 = adj1.shape[0]
    n2 = adj2.shape[0]
    I1 = np.eye(n1)
    I2 = np.eye(n2)
    W1 = np.linalg.inv(I1 - t * adj1)
    W2 = np.linalg.inv(I2 - t * adj2)
    return np.linalg.norm(W1 - W2)

# Construct graph from the dataset - nodes are connected based on k nearest neighbors
def construct_graph(dataset, k_neighbors):
    graph = nx.Graph()
    distance_matrix = cdist(dataset, dataset, metric='euclidean')

    for i in range(len(dataset)):
        neighbors = np.argsort(distance_matrix[i])[1:k_neighbors+1]
        for j in neighbors:
            graph.add_edge(i, j)

    return graph

# Define a custom distance function for k-nearest neighbors
def custom_distance(x1, x2):
    return random_walk_distance(x1, x2, t)



# Generate the swiss roll datasets
X1, y1 = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)
X2, y2 = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)

y1= np.round(y1,0)
y2= np.round(y2,0)

#y1 = np.zeros(1000)
#y1[:10] = 1

#y2 = np.zeros(1000)
#y2[:500] = 1



# Plot the datasets with labeled points
fig = plt.figure(figsize=(10, 5))

ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X1[:, 0], X1[:, 1], X1[:, 2], c=y1, alpha=0.5)
ax1.set_title('Swiss Roll Dataset 1')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(X2[:, 0], X2[:, 1], X2[:, 2], c=y2, alpha=0.5)
ax2.set_title('Swiss Roll Dataset 2')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

plt.tight_layout()
plt.show()



# Scale the input features between 0 and 1
scaler = MinMaxScaler()
X1_scaled = scaler.fit_transform(X1)
X2_scaled = scaler.fit_transform(X2)

# Construct graphs from the datasets - nodes are connected based on 3 nearest neighbors
k = 3  # Number of nearest neighbors
graph1 = construct_graph(X1_scaled, k)
graph2 = construct_graph(X2_scaled, k)

# Convert graphs to adjacency matrices
adj1 = csr_matrix(nx.to_numpy_array(graph1))
adj2 = csr_matrix(nx.to_numpy_array(graph2))


# Compute the random walk distance between the adjacency matrices
t = 0.5  # Parameter for random walk distance
rwd = random_walk_distance(adj1, adj2, t)


# Train a k-nearest neighbors classifier on dataset 1 using the custom distance function
classifier = KNeighborsClassifier(n_neighbors=5, metric=custom_distance)
classifier.fit(X2_scaled, y2)


# Classify the points in dataset 1 using the trained classifier
y1_pred = classifier.predict(X1_scaled)
#y2_pred = classifier.predict(X2_scaled)


# Calculate the accuracy
accuracy = np.mean(y1_pred == y1)
print('Accuracy:', accuracy)




# Plot the datasets before labeling
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X1[:, 0], X1[:, 1], X1[:, 2], c=y1, alpha=0.5)
ax1.set_title('Swiss Roll Dataset 1 (orginal Labeling)')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(X1[:, 0], X1[:, 1], X1[:, 2], c=y1_pred, alpha=0.5)
ax2.set_title('Swiss Roll Dataset 2 (Predicted Labeling)')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

plt.tight_layout()
plt.show()
