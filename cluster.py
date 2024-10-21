import numpy as np

# Sample 2D array (you can modify it with your data)
array = np.array(
    [
        [0, 1, 0, 0, 2],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 3, 3],
        [0, 0, 0, 0, 3],
        [1, 1, 0, 0, 0],
    ]
)


# Function to perform DFS
def dfs(array, i, j, visited):
    # Define 8-connected neighbors (can change to 4 for simpler connectivity)
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    # Stack to store the current cluster of non-zero elements
    stack = [(i, j)]
    cluster_size = 1
    # cluster = [(i, j)]

    visited[i, j] = True

    while stack:
        x, y = stack.pop()

        # Explore all neighbors
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy

            # Check if the new position is within bounds and not yet visited
            if (
                0 <= nx < array.shape[0]
                and 0 <= ny < array.shape[1]
                and not visited[nx, ny]
            ):
                # Check if the element is non-zero
                if array[nx, ny] != 0:
                    visited[nx, ny] = True
                    stack.append((nx, ny))
                    cluster_size += 1
                    # cluster.append((nx, ny))

    return cluster_size


# Function to find all clusters of non-zero elements
def find_clusters(array):
    rows, cols = array.shape
    visited = np.zeros_like(array, dtype=bool)
    clusters = []

    # Loop through the array
    for i in range(rows):
        for j in range(cols):
            # If it's a non-zero element and not visited, start DFS
            if array[i, j] != 0 and not visited[i, j]:
                cluster = dfs(array, i, j, visited)
                clusters.append(cluster)

    return clusters


# Find clusters of non-zero elements
# clusters = find_clusters(array)

# Output clusters
# for idx, cluster in enumerate(clusters):
# print(f"Cluster {idx + 1}: {cluster}")
