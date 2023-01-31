import numpy as np


class logistic_regression:
    def __init__(self):
        pass

    def sigmoid(self,x):
        """ 定义sigmoid函数 """
        return 1/(1+np.exp(-x))

    def initialize_params(self, dims):
        """ 模型参数初始化 """
        w = np.zeros((dims, 1))
        b = 0
        return w,b

    def logistic(self, X, y,W,b):
        """ 模型主体：模型计算公式，损失函数，参数的梯度公式 """
        num_train = X.shape[0]
        num_feature = X.shape[1]

        z = self.sigmoid(np.dot(X, W) + b)
        cost = -1 / num_train * np.sum(y * np.log(z) + (1-y) * np.log(1-z))

        dw = np.dot(X.T,(z-y)) / num_train
        db = np.sum(z-y) / num_train
        cost = np.squeeze(cost)
        return z, cost, db, dw
    
    def logistic_train(self, X,y, learning_rate, epochs):
        """ 基于梯度下降的参数更新训练 """
        w, b = self.initialize_params(X.shape[1])
        cost_list = []

        for i in range(1,epochs):
            z, cost, db, dw = self.logistic(X, y, w, b)
            w += w - learning_rate * dw
            b += b - learning_rate * db

            if i % 100 == 0:
                cost_list.append(cost)
                print('epoch %d cost %f'% (i,cost))
        params = {
            'w':w,
            'b':b
        }
        grads = {
            'dw':dw,
            'db':db
        }
        return cost_list, params, grads

    def predict(self, X, params):
        y_prediction = self.sigmoid(np.dot(X,params['w'])+params['b'])
        for i in range(len(y_prediction)):
            if y_prediction[i] > 0.5:
                y_prediction[i] = 1
            else:
                y_prediction[i] = 0
        return y_prediction

    def accurary(self, y_test, y_pred):
        correct_count = 0
        for i in range(len(y_test)):
            for j in range(len(y_pred)):
                if y_test[i] == y_pred[j] and i == j:
                    correct_count += 1
        accurary_score = correct_count / len(y_test)
        return accurary_score

# 创建数据
from sklearn.datasets._samples_generator import make_classification
X, labels = make_classification(n_samples = 100, n_features = 2, n_redundant = 0, n_informative = 2, n_clusters_per_class = 2)
labels = labels.reshape(-1,1)
offset = int(X.shape[0]*0.9)
X_train, y_train = X[:offset], labels[:offset]
X_test, y_test = X[offset:], labels[offset:]

model = logistic_regression()
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
cost_list, params, grads = model.logistic_train(X_train,y_train, 0.01, 1000)
print(params)

y_train_pred = model.predict(X_train, params)
accurary_score_train = model.accurary(y_train, y_train_pred)
y_test_pred = model.predict(X_test, params)
accurary_score_test = model.accurary(y_test, y_test_pred)

# 绘制决策边界
import matplotlib.pyplot as plt
def plot_logistic(X_train, y_train, params):
    n = X_train.shape[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if y_train[i] == 1:
            xcord1.append(X_train[i][0])
            ycord1.append(X_train[i][1])
        else:
            xcord2.append(X_train[i][0])
            ycord2.append(X_train[i][1])
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s = 32, c = 'red')
    ax.scatter(xcord2, ycord2, s = 32, c = 'green')
    x = np.arange(-1.5,3,0.1)
    y = (-params['b'] - params['w'][0] * x)/params['w'][1]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

plot_logistic(X_train, y_train, params)