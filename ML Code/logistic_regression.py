import numpy as np
from sklearn.datasets import make_blobs

class LogisticRegression(object):
    def __init__(self, X):
        self.learning_rate = 0.0000001
        self.iterations = 10000

        #X: mxn m: no_samples, n_no_features
        #Y: mx1 
        self.m, self.n = X.shape

    def train(self, X, Y):
        self.weights = np.random.randn(self.n, 1)
        self.bias = 0
        for it in range(self.iterations+1):
            y_pred = self.sigmoid(np.dot(X, self.weights) + self.bias)
            cost = (
            -1/
            self.m
            *np.sum(Y*np.log(y_pred) + (1-Y)*np.log(1-y_pred)))

            dW = 1/self.m*np.dot(X.T, (y_pred-Y))
            db = 1/self.m*np.sum(y_pred-Y)

            self.weights -= self.learning_rate*dW
            self.bias -= self.learning_rate*db
            if it%1000==0:
                print(f"cost at iteration {it} is {cost}")

        return self.weights, self.bias

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def predict_labels(self, X, w, b):
        y_label = self.sigmoid(np.dot(X, w) + b)
        y_pred = y_label>0.5
        return y_pred


if __name__=="__main__":
    np.random.seed(1)
    X, y = make_blobs(n_samples=1000, n_features=2)
    y = y[:, np.newaxis]

    logreg = LogisticRegression(X)
    w, b = logreg.train(X, y)
    y_pred = logreg.predict_labels(X, w, b)
    print(f'Accuracy is {np.sum(y_pred==y)/X.shape[0]}')

