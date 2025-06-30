import random
import math
import os
import matplotlib.pyplot as plt

def euclidean_distance_squared(point1, point2):
    return sum((x - y)**2 for x, y in zip(point1, point2))

def mean_point(points, dimension):
    if not points:
        return [0] * dimension
    num_points = len(points)
    sums = [0] * dimension
    for point in points:
        for i in range(dimension):
            sums[i] += point[i]
    return [total / num_points for total in sums]

def kmeans(data, k, max_iters=100, output_dir="kmeans_nd_output"):
    os.makedirs(output_dir, exist_ok=True)

    dimension = len(data[0])
    centroids = random.sample(data, k)
    losses = []

    print(f"Initial centroids: {centroids}")

    for iteration in range(max_iters):
        # Assignment step
        clusters = [[] for _ in range(k)]
        for point in data:
            distances = [euclidean_distance_squared(point, c) for c in centroids]
            closest_idx = distances.index(min(distances))
            clusters[closest_idx].append(point)

        # Compute WCSS loss
        loss = 0
        for i in range(k):
            loss += sum(euclidean_distance_squared(point, centroids[i]) for point in clusters[i])
        losses.append(loss)

        print(f"Iteration {iteration+1}, Loss={loss:.4f}")
        for idx, cluster in enumerate(clusters):
            print(f"  Cluster {idx}: {len(cluster)} points, Centroid={centroids[idx]}")

        # Update centroids
        old_centroids = centroids.copy()
        for i in range(k):
            if clusters[i]:
                centroids[i] = mean_point(clusters[i], dimension)

        # Check for convergence
        if centroids == old_centroids:
            print("Converged.")
            break

    # Loss curve
    plt.figure()
    plt.plot(range(1, len(losses)+1), losses, marker='o')
    plt.title("Loss vs Iteration (k-means in N-D)")
    plt.xlabel("Iteration")
    plt.ylabel("Total WCSS")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()

    return centroids, clusters, losses

# Example dataset: 5D
data_5d = [
    [1, 1, 1, 1, 1], [1.2, 0.9, 1.1, 0.8, 1.0], [0.8, 1.1, 0.9, 1.2, 1.1],
    [5, 5, 5, 5, 5], [5.1, 4.9, 5.2, 5.3, 4.8], [4.9, 5.2, 5.1, 4.8, 5.0],
    [9, 1, 8, 9, 1], [9.1, 1.2, 8.2, 8.9, 1.1], [8.8, 0.9, 7.9, 9.2, 0.8]
]

final_centroids, final_clusters, losses = kmeans(data_5d, k=3)

print("\nFinal clusters and centroids:")
for idx, cluster in enumerate(final_clusters):
    print(f"Cluster {idx}: {len(cluster)} points, Centroid={final_centroids[idx]}")
