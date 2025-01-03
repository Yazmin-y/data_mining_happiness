import numpy as np


def KNN(train_x, train_y, test_x, k=3):
    train_y = train_y.values
    num_train = train_x.shape[0]
    num_test = test_x.shape[0]
    dist = np.zeros((num_test, num_train))
    for i in range(num_test):
        dist[i] = np.reshape(np.sqrt(np.sum(np.square(test_x[i] - train_x), axis=1)), [1, num_train])
    predictedLabels = np.zeros((num_test, 1))
    for i in range(num_test):
        close_k = train_y[np.argsort(dist[i])[:k]].astype(np.int)
        predictedLabels[i] = np.argmax(np.bincount(close_k))
    return predictedLabels