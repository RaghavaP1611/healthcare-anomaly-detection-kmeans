import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# Load dataset
df = pd.read_csv("sample_submission_data.csv")

# Select feature for clustering
X = df[["submission_count"]]

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
df["cluster"] = kmeans.fit_predict(X_scaled)

# Plot clustering results
plt.figure(figsize=(10, 5))
plt.scatter(df.index, df["submission_count"], c=df["cluster"])

plt.title("Healthcare Submission Anomaly Detection")
plt.xlabel("Submission Record")
plt.ylabel("Submission Count")

# Save visualization
plt.savefig("anomaly_detection_output.png")

print(df)
print("\nClustering completed successfully.")
print("Visualization saved as anomaly_detection_output.png")
