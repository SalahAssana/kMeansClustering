import random
import matplotlib.pyplot as plt
import math
import os

def euclidean_distance_squared(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def mean_point(points):
    if not points:
        return (0, 0)
    x_mean = sum(p[0] for p in points) / len(points)
    y_mean = sum(p[1] for p in points) / len(points)
    return (x_mean, y_mean)

def kmeans_2d(data, k, max_iters=100, output_dir="kmeans2d_output"):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Initialize centroids by randomly picking k points
    centroids = random.sample(data, k)
    print(f"Initial centroids: {centroids}")

    losses = []

    for iteration in range(max_iters):
        # 2. Assignment step
        clusters = [[] for _ in range(k)]
        for point in data:
            distances = [euclidean_distance_squared(point, c) for c in centroids]
            closest_idx = distances.index(min(distances))
            clusters[closest_idx].append(point)

        # 3. Compute loss (WCSS)
        loss = 0
        for i in range(k):
            loss += sum(euclidean_distance_squared(point, centroids[i]) for point in clusters[i])
        losses.append(loss)

        print(f"Iteration {iteration+1}, Loss={loss:.4f}")
        for idx, cluster in enumerate(clusters):
            print(f"  Cluster {idx}: {len(cluster)} points, Centroid={centroids[idx]}")

        # 4. Plot clusters
        colors = ['red', 'green', 'blue', 'purple', 'orange']
        plt.figure()
        for idx, cluster in enumerate(clusters):
            if cluster:
                x_vals = [p[0] for p in cluster]
                y_vals = [p[1] for p in cluster]
                plt.scatter(x_vals, y_vals, color=colors[idx], label=f'Cluster {idx}')
            plt.scatter(*centroids[idx], color=colors[idx], s=200, marker='x')
        plt.title(f"Iteration {iteration+1}")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"clusters_iter_{iteration+1}.png"))
        plt.close()

        # 5. Update centroids
        old_centroids = centroids.copy()
        for i in range(k):
            if clusters[i]:
                centroids[i] = mean_point(clusters[i])

        # 6. Check for convergence
        if centroids == old_centroids:
            print("Converged.")
            break

    # Plot loss curve
    plt.figure()
    plt.plot(range(1, len(losses)+1), losses, marker='o')
    plt.title("Loss vs Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Total WCSS")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()

    return centroids, clusters, losses

# Example usage
data_2d = [
    # Cluster-ish near (1,1)
    (1, 2), (2, 1.5), (1.5, 1.8), (1.8, 1), (2,2), (1.2,1.3),
    # Cluster-ish near (4,4) but more spread out
    (4, 4), (4.5, 3.8), (5, 4.2), (3.5, 4.5), (4.2,3.7), (5,5), (3.8,3.9),
    # Cluster-ish near (8,1) but mixed
    (8,1), (7.5, 1.2), (8.2, 0.8), (7.8, 1.5), (8.5,1.3), (8,2), (7.7,0.9)
]

final_centroids, final_clusters, losses = kmeans_2d(data_2d, k=3)

print("\nFinal clusters and centroids:")
for idx, cluster in enumerate(final_clusters):
    print(f"Cluster {idx}: {len(cluster)} points, Centroid={final_centroids[idx]}")
