import random
import matplotlib.pyplot as plt
import os

def kmeans_1d_with_loss_savefigs(data, k, max_iters=100, output_dir="kmeans_output"):
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 1. Initialize centroids randomly
    centroids = random.sample(data, k)
    print(f"Initial centroids: {centroids}")

    losses = []

    for iteration in range(max_iters):
        # 2. Assignment step
        clusters = [[] for _ in range(k)]
        for point in data:
            distances = [abs(point - c) for c in centroids]
            closest_idx = distances.index(min(distances))
            clusters[closest_idx].append(point)

        # 3. Compute loss (WCSS)
        loss = 0
        for i in range(k):
            loss += sum((point - centroids[i])**2 for point in clusters[i])
        losses.append(loss)

        print(f"Iteration {iteration+1}, Loss={loss:.4f}")
        for idx, cluster in enumerate(clusters):
            print(f"  Cluster {idx}: Points={cluster}, Centroid={centroids[idx]:.3f}")

        # 4. Plot clusters on number line and save figure
        colors = ['red', 'green', 'blue', 'purple', 'orange']
        plt.figure(figsize=(8, 2))
        for idx, cluster in enumerate(clusters):
            plt.scatter(cluster, [0]*len(cluster), color=colors[idx], label=f'Cluster {idx}')
            plt.scatter(centroids[idx], 0, color=colors[idx], s=200, marker='x')  # removed edgecolor
        plt.title(f"Iteration {iteration+1}: Clusters")
        plt.yticks([])
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"clusters_iter_{iteration+1}.png"))
        plt.close()

        # 5. Update centroids
        old_centroids = centroids.copy()
        for i in range(k):
            if clusters[i]:
                centroids[i] = sum(clusters[i]) / len(clusters[i])

        # 6. Check for convergence
        if centroids == old_centroids:
            print("Converged.")
            break

    # Finally plot the loss curve
    plt.figure()
    plt.plot(range(1, len(losses)+1), losses, marker='o')
    plt.title("Loss vs Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Total within-cluster sum of squares (WCSS)")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()

    return centroids, clusters, losses

# Example usage
data = [1.0, 2.1, 1.9, 5.0, 6.1, 5.9, 8.0, 8.2, 9.0]
final_centroids, final_clusters, losses = kmeans_1d_with_loss_savefigs(data, k=3)

print("\nFinal clusters and centroids:")
for idx, cluster in enumerate(final_clusters):
    print(f"Cluster {idx}: Points={cluster}, Centroid={final_centroids[idx]:.3f}")
