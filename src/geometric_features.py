import numpy as np
from sklearn.neighbors import NearestNeighbors

def compute_geometric_features(points, k=20):
    """
    Compute geometric features for each point in the point cloud
    
    Args:
        points: numpy array of shape (N, 3) containing point coordinates
        k: number of nearest neighbors to use
    
    Returns:
        features: numpy array of shape (N, F) containing geometric features
        normals: numpy array of shape (N, 3) containing surface normals
    """
    # Find k nearest neighbors for each point
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    # Initialize arrays for features
    N = points.shape[0]
    normals = np.zeros((N, 3))
    curvatures = np.zeros(N)
    densities = np.zeros(N)
    heights = np.zeros(N)
    
    for i in range(N):
        # Get neighborhood points
        neighborhood = points[indices[i]]
        
        # Compute covariance matrix
        centered = neighborhood - np.mean(neighborhood, axis=0)
        cov = np.dot(centered.T, centered) / (k - 1)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Surface normal (smallest eigenvector)
        normals[i] = eigenvectors[:, 0]
        
        # Curvature estimation
        curvatures[i] = eigenvalues[0] / np.sum(eigenvalues)
        
        # Local point density
        densities[i] = np.mean(distances[i, 1:])  # Exclude self
        
        # Height feature (z-coordinate relative to local neighborhood)
        heights[i] = points[i, 2] - np.mean(neighborhood[:, 2])
    
    # Combine all features
    features = np.column_stack([
        curvatures,
        densities,
        heights,
        normals
    ])
    
    return features, normals