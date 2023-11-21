import numpy as np
import matplotlib.pyplot as plt

# Load the training dataset and process the labels
train_data = np.genfromtxt('data/dataset1.csv', delimiter=';', usecols=[1, 2, 3, 4, 5, 6, 7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
train_dates = np.genfromtxt('data/dataset1.csv', delimiter=';', usecols=[0])
train_labels = []
for label in train_dates:
    if label < 20000301:
        train_labels.append('winter')
    elif 20000301 <= label < 20000601:
        train_labels.append('spring')
    elif 20000601 <= label < 20000901:
        train_labels.append('summer')
    elif 20000901 <= label < 20001201:
        train_labels.append('autumn')
    else:  # from 01-12 to end of year
        train_labels.append('winter')

# Load the test / validation dataset and process the labels
val_data = np.genfromtxt('data/validation1.csv', delimiter=';', usecols=[1, 2, 3, 4, 5, 6, 7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
val_dates = np.genfromtxt('data/validation1.csv', delimiter=';', usecols=[0])
val_labels = []
for label in val_dates:
    if label < 20010301:
        val_labels.append('winter')
    elif 20010301 <= label < 20010601:
        val_labels.append('spring')
    elif 20010601 <= label < 20010901:
        val_labels.append('summer')
    elif 20010901 <= label < 20011201:
        val_labels.append('autumn')
    else:  # from 01-12 to end of year
        val_labels.append('winter')

days_data = np.genfromtxt('data/days.csv', delimiter=';', usecols=[1, 2, 3, 4, 5, 6, 7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})

def initialize_centers(train_data, k):
    # Randomly initialize k cluster centers from the data points
    np.random.seed(0)  # Set seed for reproducibility
    centers = train_data[np.random.choice(train_data.shape[0], size=k, replace=False)]
    return centers

def kmeans(X_train_data, K_centroids):
    cluster_assignments = []
    wcss = 0
    for x_train_feature in X_train_data:
        distances = [np.linalg.norm(center - x_train_feature) for center in K_centroids]
        cluster_assignments.append(np.argmin(distances))

    cluster_assignments = np.array(cluster_assignments)  # Convert clusters list to NumPy array
    
    
    for i, center in enumerate(K_centroids):
        cluster_points = X_train_data[cluster_assignments == i]
        wcss += np.sum((cluster_points - center) ** 2)

    # Update cluster centers
    new_centers = np.array([X_train_data[cluster_assignments == center].mean(axis=0) for center in range(len(K_centroids))])
    
    return new_centers, cluster_assignments, wcss

# ## Option 1: uncomment for normalization (less accurate)
# # Apply normalization
# X_train_data_normalized = (train_data - train_data.min(axis=0)) / (train_data.max(axis=0) - train_data.min(axis=0))
# val_data_normalized = (val_data - val_data.min(axis=0)) / (val_data.max(axis=0) - val_data.min(axis=0))
# wcss_values = []
# k_values = range(1, 8)
# # Apply K-means clustering
# for k in k_values:
#     init_centroids = initialize_centers(train_data, k)
#     cluster_centers = init_centroids
#     for tries in range(100):
#         new_cluster_centers, cluster_assignments, wcss = kmeans(X_train_data_normalized, cluster_centers)
#         cluster_centers = new_cluster_centers
#     wcss_values.append(wcss)
# ##

## Option 2: uncomment for no normalization (more accurate)
wcss_values = []
k_values = range(1, 8)
# Apply K-means clustering
for k in k_values:
    init_centroids = initialize_centers(train_data, k)
    cluster_centers = init_centroids
    for tries in range(100):
        new_cluster_centers, cluster_assignments, wcss = kmeans(train_data, cluster_centers)
        cluster_centers = new_cluster_centers
    wcss_values.append(wcss)
## 

# Map cluster indexes to actual labels
cluster_labels = {}
for cluster_idx in range(len(cluster_centers)):
    cluster_indexes = np.where(cluster_assignments == cluster_idx)[0]
    cluster_actual_labels = [train_labels[idx] for idx in cluster_indexes]
    most_common_label = max(set(cluster_actual_labels), key=cluster_actual_labels.count)
    cluster_labels[cluster_idx] = most_common_label

# Calculate accuracy based on cluster labels and true labels
correct_predictions = sum(1 for idx, true_label in enumerate(train_labels) if cluster_labels[cluster_assignments[idx]] == true_label)
total_samples = len(train_labels)
accuracy = (correct_predictions / total_samples) * 100

print("Cluster centers:")
print(cluster_centers)
for cluster_idx, label in cluster_labels.items():
    print(f"Cluster {cluster_idx} corresponds to {label}")
print("\nCluster assignments:")
print(cluster_assignments)
print(f"Accuracy of K-means clustering on training data: {accuracy:.2f}%")

plt.figure(figsize=(8, 5))
plt.plot(k_values, wcss_values, marker='o', linestyle='-', color='b')
plt.title('Scree Plot / Elbow Method')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.xticks(k_values)
plt.grid(True)
plt.show()