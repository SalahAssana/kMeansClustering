import random
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D  # triggers 3D support

def euclidean_distance_squared_3d(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2

def mean_point_3d(points):
    if not points:
        return (0, 0, 0)
    x_mean = sum(p[0] for p in points) / len(points)
    y_mean = sum(p[1] for p in points) / len(points)
    z_mean = sum(p[2] for p in points) / len(points)
    return (x_mean, y_mean, z_mean)

def kmeans_3d(data, k, max_iters=100, output_dir="kmeans3d_output"):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Initialize centroids randomly
    centroids = random.sample(data, k)
    print(f"Initial centroids: {centroids}")

    losses = []
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'cyan', 'magenta']

    for iteration in range(max_iters):
        # 2. Assignment step
        clusters = [[] for _ in range(k)]
        for point in data:
            distances = [euclidean_distance_squared_3d(point, c) for c in centroids]
            closest_idx = distances.index(min(distances))
            clusters[closest_idx].append(point)

        # 3. Compute WCSS loss
        loss = 0
        for i in range(k):
            loss += sum(euclidean_distance_squared_3d(point, centroids[i]) for point in clusters[i])
        losses.append(loss)

        print(f"Iteration {iteration+1}, Loss={loss:.4f}")
        for idx, cluster in enumerate(clusters):
            print(f"  Cluster {idx}: {len(cluster)} points, Centroid={centroids[idx]}")

        # 4. Plot 3D scatter and save
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for idx, cluster in enumerate(clusters):
            if cluster:
                x_vals = [p[0] for p in cluster]
                y_vals = [p[1] for p in cluster]
                z_vals = [p[2] for p in cluster]
                ax.scatter(x_vals, y_vals, z_vals, color=colors[idx % len(colors)], label=f'Cluster {idx}')
            # plot centroid
            ax.scatter(*centroids[idx], color=colors[idx % len(colors)], s=200, marker='x')
        ax.set_title(f"Iteration {iteration+1}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        plt.savefig(os.path.join(output_dir, f"clusters_iter_{iteration+1}.png"))
        plt.close()

        # 5. Update centroids
        old_centroids = centroids.copy()
        for i in range(k):
            if clusters[i]:
                centroids[i] = mean_point_3d(clusters[i])

        # 6. Check for convergence
        if centroids == old_centroids:
            print("Converged.")
            break

    # 7. Plot loss curve
    plt.figure()
    plt.plot(range(1, len(losses)+1), losses, marker='o')
    plt.title("Loss vs Iteration (3D k-means)")
    plt.xlabel("Iteration")
    plt.ylabel("Total WCSS")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()

    return centroids, clusters, losses

# Example 3D dataset
data_3d = [
    # cluster near (1,1,1)
    (0.0, 1.5, 0.8), (2.0, 0.5, 1.2), (1.2, 2.4, 0.9), 
    (0.8, 1.2, 2.3), (1.3, -0.1, 0.7), (-0.5, 1.0, 1.0),
    (1.9, 1.3, 1.2), (0.7, 2.8, 1.1), (2.2, 1.1, 2.4),
    (1.0, -0.9, 0.8), (1.3, 1.2, -0.9), (0.9, 2.1, 1.3),
    (1.1, -0.8, 1.0), (0.8, 0.0, 1.1), (2.2, 1.3, 1.0),

    # cluster near (7,7,7)
    (7.5, 7.1, 6.8), (6.2, 6.4, 7.2), (7.8, 7.3, 7.1),
    (6.1, 8.0, 7.4), (8.4, 6.8, 6.9), (7.0, 8.4, 7.0),
    (6.7, 5.8, 7.2), (7.9, 7.1, 6.7), (6.2, 7.0, 8.3),
    (7.0, 6.9, 7.1), (6.1, 7.2, 6.8), (8.7, 6.0, 7.2),
    (7.3, 7.1, 6.9), (6.0, 7.3, 7.0), (8.1, 6.9, 7.1),

    # cluster near (14,2,12)
    (14.5, 1.2, 12.0), (13.9, 14.8, 11.7), (14.3, 1.1, 12.2),
    (13.8, 3.3, 11.9), (14.0, 13.9, 12.3), (14.2, 2.0, 12.1),
    (14.1, 3.3, 11.8), (13.9, 14.0, 12.0), (14.4, 0.9, 12.2),
    (14.0, 1.1, 11.9), (14.2, 14.8, 12.1), (13.7, 3.2, 12.0),
    (14.3, 2.0, 11.7), (13.8, 13.9, 12.3), (14.1, 1.1, 11.8),
]

final_centroids, final_clusters, losses = kmeans_3d(data_3d, k=3)

print("\nFinal clusters and centroids:")
for idx, cluster in enumerate(final_clusters):
    print(f"Cluster {idx}: {len(cluster)} points, Centroid={final_centroids[idx]}")
