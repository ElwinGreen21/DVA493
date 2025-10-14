# Assignment 3: Unsupervised Learning
import numpy as np
import os

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





def main():
    classes, data = file_reader()
    data = normalize_data(data)
    cov = compute_covariance(data)
    new_features, eigenvalues, eigenvectors = compute_pca(cov, data)

    # Test print: first row
    print(f"Klass: {classes[0]}")
    print("Normalized Features:", " ".join(f"{x:.4f}" for x in data[0]))
    print("PCA Features:", " ".join(f"{x:.4f}" for x in new_features[0]))


if __name__ == "__main__":
    main()
