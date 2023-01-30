import numpy as np
#W:nx1  X:nxm  y:1xm
#m: training examples n: feature size
class LinearRegression(object):
    def __init__(self):
        self.learning_rate = 0.0001
        self.iteration = 10000
    def y_hat(self, W, X):
        return np.dot(W.T, X)
    def cost(self, Y_hat, Y):
       J =  1/self.m*np.sum(np.power(Y_hat-Y, 2))
       return J
    def gradientDescent(self, W, X, Y_hat, Y):
        dLdW = 2/self.m*np.dot(X, (Y_hat-Y).T)
        W = W - self.learning_rate*dLdW
        return W

    def main(self, X, Y):
        ones = np.ones((1, X.shape[1]))
        X = np.append(ones, X, axis=0)

        self.n = X.shape[0]
        self.m = X.shape[1]

        w = np.random.rand(self.n, 1)

        for i in range(self.iteration):
            y_hat = self.y_hat(w, X)
            j = self.cost(y_hat, Y)

            if i%1000==0:
                print(f"cost at iteration {i} is {j}")
            w = self.gradientDescent(w,X,y_hat,Y)

        return w



if __name__=="__main__":
    X = np.random.rand(1, 500)
    Y = 5*X + np.random.rand(1, 500)*0.1
    regression = LinearRegression()
    w = regression.main(X, Y)





