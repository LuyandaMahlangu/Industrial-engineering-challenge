import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

names = "C:/Users/Small/Downloads/query_result.csv"
df = pd.read_csv(names)

df.head()

# Extract latitude and longitude for clustering
locations = df[["latitude", "longitude"]]

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
df["cluster"] = kmeans.fit_predict(locations)

# Visualize the clusters
plt.figure(figsize=(10, 6))
for cluster in range(5):
    cluster_data = df[df["cluster"] == cluster]
    plt.scatter(cluster_data["longitude"], cluster_data["latitude"], label=f"Cluster {cluster}")

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c="red", marker="X", s=200, label="Centroids")
plt.title("Emergency Incident Clusters")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.show()

# Save the clustered data
output = "C:/Users/Small/Downloads/clustered_emergency_data.csv"
df.to_csv(output, index=False)