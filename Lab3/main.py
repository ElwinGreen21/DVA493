# Assignment 3: Unsupervised Learning
import numpy as np
import os
import matplotlib.pyplot as plt

ROWS = 178
COLS = 13

os.chdir(os.path.dirname(__file__))

def file_reader():
    classes = np.zeros(ROWS, dtype=int)
    data = np.zeros((ROWS, COLS), dtype=float)

    with open("WINE.txt", "r") as fp:
        for i, line in enumerate(fp):
            values = line.strip().split()
            if len(values) != COLS + 1:
                print(f"Fel vid läsning av rad {i+1}")
                continue

            classes[i] = int(values[0])
            data[i] = np.array(values[1:], dtype=float)

    return classes, data


def normalize_data(data):
    # Compute mean and standard deviation for each column
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    
    # Avoid division by zero in case std is 0
    std[std == 0] = 1.0
    
    # Standardize the data
    normalized_data = (data - mean) / std
    return normalized_data



def compute_covariance(data):
    cov = np.zeros((COLS, COLS))
    for i in range(COLS):
        for j in range(COLS):
            cov[i, j] = np.sum(data[:, i] * data[:, j]) / (ROWS - 1)
    return cov


def compute_pca(cov, data):
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    idx = np.argsort(eigenvalues)[::-1] 
    eigenvalues = eigenvalues[idx] 
    eigenvectors = eigenvectors[:, idx]

    new_features = np.dot(data, eigenvectors)

    return new_features, eigenvalues, eigenvectors

def kmeans(data, k=3, max_iters=100, tol=1e-4):
    np.random.seed(42)  
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]

    for iteration in range(max_iters):
        # Step 1: Assign each point to the nearest centroid
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)  # Shape: (n_samples, k)
        labels = np.argmin(distances, axis=1)
        
        # Step 2: Update centroids as the mean of assigned points
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        
        # Step 3: Check for convergence
        if np.all(np.linalg.norm(new_centroids - centroids, axis=1) < tol):
            break
        
        centroids = new_centroids
    
    return labels, centroids




def main():
    classes, data = file_reader()
    data = normalize_data(data)
    cov = compute_covariance(data)
    new_features, eigenvalues, eigenvectors = compute_pca(cov, data)

    k = 3

    
    for n_components in [2, 5, 8]:
        reduced_data = new_features[:, :n_components]

        labels, centroids = kmeans(reduced_data, k=k)

        
        cluster_sizes = np.bincount(labels)

        
        print(f"\nK-means with top {n_components} principal components:")
        for i in range(k):
            print(f"  Cluster {i+1}: {cluster_sizes[i]} points")
        print(f"  Centroid shape: {centroids.shape}")  

        
        if n_components == 2:
            plt.figure(figsize=(7, 5))
            for cluster_id in range(k):
                cluster_points = reduced_data[labels == cluster_id]
                plt.scatter(
                    cluster_points[:, 0],
                    cluster_points[:, 1],
                    label=f"Cluster {cluster_id+1}",
                    alpha=0.7
                )

            # Plot centroids
            plt.scatter(
                centroids[:, 0],
                centroids[:, 1],
                c="black",
                marker="X",
                s=200,
                label="Centroids"
            )

            plt.title("K-means Clustering (Top 2 Principal Components)")
            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")
            plt.legend()
            plt.grid(True)
            plt.show()


if __name__ == "__main__":
    main()
