import numpy as np

class LinearRegression:

    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters= n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        #init parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            yhat = np.dot(X, self.weights) + self.bias

            dw = (1/n_samples) * np.dot(X.T, (yhat - y))
            db = (1/n_samples) * np.sum(yhat - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted

if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    predicted = regressor.predict(X_test)

    def mse(y_true, y_pred):
        return np.mean((y_true-y_pred)**2)
    
    mse_value = mse(y_test, predicted)
    print(mse_value)

    y_pred_line = regressor.predict(X)
    plt.figure(figsize=(14,7))
    cmap = plt.get_cmap('viridis')
    plt.scatter(X_train, y_train, color=cmap(0.2), s=10)
    plt.scatter(X_test, y_test, color=cmap(0.6), s=10)
    plt.plot(X, y_pred_line, color='black', linewidth=2, label='Prediction')
    plt.show()