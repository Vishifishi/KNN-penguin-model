import numpy as np
import matplotlib.pyplot as plt


class KNearestNeighbors():
    def __init__(self, k):
        self.k = k
    def train(self, X, y):
        self.X_train = X
        self.y_train = y
    def predict(self, X_test, num_loops = 0):
        if num_loops == 0:
            distances = self.compute_distance.vectorized(X_test)
        elif num_loops == 1:
            distances = self.compute_distance_one_loop(X_test)
        else:
            distances = self.compute_distance_two_loops(X_test)

        return self.predict(distances)


    def compute_distance_two_loops(self, X_test):
        num_test = X_test.shape[0]
        num_train = self.X_train.shape[0]
        distances = np.zeros((num_test, num_train))

        for i in range(num_test):
            for j in range(num_train):
                distances[i, j] = np.sqrt(np.sum((X_test[i] - self.X_train[j]) ** 2))
        return distances

    def compute_distance_one_loop(self, X_test):
        num_test = X_test.shape[0]
        num_train = self.X_train.shape[0]
        distances = np.zeros((num_test, num_train))

        for i in range(num_test):
            distances[i, :] = np.sqrt(np.sum((X_test[i] - self.X_train) ** 2, axis=1))
        return distances

    def compute_distance_vectorized(self, X_test):
        X_test_squared = np.sum(X_test ** 2, axis=1, keepdims=True)
        X_train_squared = np.sum(self.X_train ** 2, axis=1, keepdims=True)
        two_X_test_X_train = np.dot(X_test, self.X_train.T)

        return np.sqrt(
            self.eps + X_test_squared - 2 * two_X_test_X_train + X_train_squared.T
        )

    def predict_labels(self, distances):
        num_test = distances.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):

            closest_y_indices = np.argsort(distances[i])[:self.k]
            closest_y = self.y_train[closest_y_indices]
            y_pred[i] = np.bincount(closest_y).argmax()

        return y_pred

if __name__ == "__main__":
    X = np.loadtxt("example_data/data.txt", delimiter=",")
    y = np.loadtxt("example_data/targets.txt")

    X = np.array([[1, 1], [3, 1], [1, 4], [2, 4], [3, 3], [5, 1]])
    y = np.array([0, 0, 0, 1, 1, 1])

    KNN = KNearestNeighbor(k=1)
    KNN.train(X, y)
    y_pred = KNN.predict(X, num_loops=0)
    print(f"Accuracy: {sum(y_pred == y) / y.shape[0]}")