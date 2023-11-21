import numpy as np

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

def knn_fit(X_train_data, Y_train_labels, initial_k=3):
    """
    Fits a k-nearest neighbors (KNN) model by storing the training data for later prediction.
    Parameters:
    X_train_data : array-like, shape (n_samples, n_features)
        Training data features containing feature vectors of examples.
    Y_train_labels : array-like, shape (n_samples,)
        Training data labels corresponding to the examples.
    k : int, optional (default=3)
        Number of neighbors to consider in the KNN algorithm.

    Returns:
    predict : function
        Function that predicts labels for test data based on the fitted KNN model.
    """
    # Apply Min-Max scaling for normalization on train data axis=0 means for each column in the feature
    X_train_data_normalized = (X_train_data - X_train_data.min(axis=0)) / (X_train_data.max(axis=0) - X_train_data.min(axis=0))

    def predict(X_test_data):
        """
        Predicts the class labels for the given test data using the fitted KNN model.

        Parameters:
        X_test_data : array-like, shape (n_samples, n_features)
            Test data features for which class labels need to be predicted.

        Returns:
        predictions : array-like, shape (n_samples,)
            Predicted class labels for the test data.
        """
        # Apply Min-Max scaling for normalization on test data, axis=0 means for each column in the feature
        X_test_data_normalized = (X_test_data - X_train_data.min(axis=0)) / (X_train_data.max(axis=0) - X_train_data.min(axis=0))

        predictions = []
        for x_test_feature in X_test_data_normalized:
          # Calculates the euclidean distance of the test feature to all the points in training data
          distances = [np.linalg.norm(x_train_feature - x_test_feature) for x_train_feature in X_train_data_normalized]
          # Sorts the indexes in ascending distance order and add the labels of the first temp_k-indexes to k_nearest_labels
          k_indices = np.argsort(distances)[:initial_k]
          k_nearest_labels = [Y_train_labels[i] for i in k_indices]
          # Creates a set to rid duplicates and uses max the count the occurances for each element in the set
          # In case of a tie it selects the nearest neighbor (tried decreasing K in case of tie, but led to more inaccuracy)
          most_common = max(set(k_nearest_labels), key=k_nearest_labels.count)
          predictions.append(most_common)
        return predictions  # Returning the predicted labels for the test data
    return predict  # Returning the predict function for later use

X_train_data = train_data
Y_train_labels = train_labels
highest_acc = 0
highest_acc_k = 1

for k in range(1, 30):
    # Fit the KNN model
    predict = knn_fit(X_train_data, Y_train_labels, initial_k=k)
    # Use the previously fitted KNN model to predict labels for validation data
    predictions_val = predict(val_data)
    predictions_days = predict(days_data)

    # Compare predicted labels with actual labels
    correct_predictions = sum(1 for true_label, predicted_label in zip(val_labels, predictions_val) if true_label == predicted_label)
    total_samples = len(val_labels)
    accuracy = correct_predictions / total_samples * 100

    if accuracy > highest_acc:
        highest_acc = accuracy
        highest_acc_k = k

    # Print accuracy of the model on the validation data
    print(f'Prediction with k = "{k}"')
    print(f'Days prediction {predictions_days}')
    print(f'Accuracy on validation data: {accuracy:.2f}%')
    print()
print(f'Highest accuracy: {highest_acc:.2f}%, with k: {highest_acc_k}')
