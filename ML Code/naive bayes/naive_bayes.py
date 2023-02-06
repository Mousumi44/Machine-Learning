import numpy as np

"""
:X: num_examples, num_features
:Y: num_examples
"""

class NaiveBayes(object):
    def __init__(self, X, Y):
        self.num_examples, self.num_features = X.shape
        self.num_classes = len(np.unique(Y))
        self.eps = 1e-6

    def fit(self, X):
        self.clases_mean = {}
        self.clases_var = {}
        self.classes_prior = {}

        for c in range(self.num_classes):
            X_c = X[Y==c]
            self.clases_mean[str(c)] = np.mean(X_c, axis=0)
            self.clases_var[str(c)] = np.var(X_c, axis=0)
            self.classes_prior[str(c)] = X_c.shape[0]/X.shape[0]

    def density_function(self, x, mean, sigma):
        # Calculate probability from Gaussian density function
        const = -self.num_features / 2 * np.log(2 * np.pi) - 0.5 * np.sum(
            np.log(sigma + self.eps)
        )
        probs = 0.5 * np.sum(np.power(x - mean, 2) / (sigma + self.eps), 1)
        return const - probs

    def predict(self, X):
        probs = np.zeros((self.num_examples, self.num_classes))
        for c in range(self.num_classes):

            prior_c = self.classes_prior[str(c)]
            probs_c = self.density_function(X, self.clases_mean[str(c)], self.clases_var[str(c)])
            probs[:, c] = probs_c + np.log(prior_c)
        return np.argmax(probs, axis=1)
        




if __name__=="__main__":
    X = np.loadtxt('data.txt', delimiter=',')
    Y = np.loadtxt('targets.txt')-1

    NB =NaiveBayes(X,Y)
    NB.fit(X)
    y_pred = NB.predict(X)

    print(f'accuracy: {sum(y_pred==Y)/X.shape[0]}')
    