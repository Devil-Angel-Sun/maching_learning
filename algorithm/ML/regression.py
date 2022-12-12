import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

## 定义数据
from sklearn.datasets import load_diabetes
from sklearn.utils import shuffle
data, target = load_diabetes().data, load_diabetes().target
X,y = shuffle(data, target, random_state=11)
offset = int(X.shape[0]*0.8)
## 划分数据集
X_train, y_train = X[:offset], y[:offset].reshape((-1,1))
X_test, y_test = X[offset:], y[offset:].reshape((-1,1))

## 线性回归

class lr_model():
    def __init__(self):
        pass

    def initialize_params(self, dims):
        w = np.zeros((dims, 1))
        b = 0
        return w,b

    def linear_loss(self, X, y, w, b):
        num_train = X.shape[0]
        num_fearture = X.shape[1]
        y_hat = np.dot(X,w)+b
        loss = np.sum((y_hat-y)^2)/num_train
        dw = np.dot(X.T, (y_hat-y))/num_train
        db = np.sum((y_hat-y))/num_train
        return y_hat, loss, dw, db

    def linear_train(self, X,y,learning_rate, epochs):
        w,b = self.initialize_params(X.shape[1])
        for i in range(1,epochs):
            y_hat, loss, dw, db = self.linear_loss(X, y, w, b)  
            w += -learning_rate * dw
            b += -learning_rate * db
            if i % 1000 == 0:
                print('epoch %d loss %f'%(i,loss))

            params = {
                "w":w,
                "b":b
            }

            grads = {
                "dw":dw,
                "db":db
            }
        return loss, params, grads
        
    def predict(self, X,params):
        w = params['w']
        b = params['b']
        y_pred = np.dot(X, w)+b
        return y_pred
    
lr = lr_model()
loss, loss_list, params, greds = lr.linear_train(X_train,y_train, 0.01, 3000)
print(params)
y_pred = lr.predict(X_test, params)
print(r2_score(y_test, y_pred))

# sklearn 实现
reg = LinearRegression()
reg.fit(X_train,y_train)
y_predict=reg.predict(X_test)
print("mean_square_error:%.2f"%mean_squared_error(y_test,y_predict))
print('Coefficient of determination: %.2f'% r2_score(y_test, y_predict))
print("coefficient of the model:\n{}".format(reg.coef_))
# Plot outputs
plt.scatter(X_test, y_test,  color='black')


#################################################################################
## Lasso 回归
class Lasso():
    def __init__(self):
        pass

    def initialize_params(self, dims):
        """初始化参数"""
        w = np.zeros((dims,1))
        b = 0
        return w,b
    
    def sign(self, x):
        """定义符号函数并进行向量化,用于对L1正则化项的梯度计算"""
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0
    
    def l1_norm(self,X,y,w,b,alpha):
        """定义Lasso损失函数"""
        num_train = X.shape[0]
        num_feature = y.shape[1]
        y_hat = np.dot(X,w)+b
        loss = np.sum((y_hat-y)**2)+np.sum(alpha*abs(w))
        dw = np.dot(X.T,(y_hat-y))/num_train+alpha*np.vectorize(self.sign)(w) # np.vectorize()对符号函数进行向量化
        db = np.sum((y-y_hat))/num_train
        return y_hat, loss, dw, db
    
    def lasso_train(self,X, y, learning_rate, epochs):
        """定义 lasso 训练函数"""
        loss_list = []
        w, b = self.initialize_params(X.shape[1])
        for i in range(1,epochs):
            y_hat, loss, dw, db = self.l1_norm(X,y,w,b,0.1)
            w += -learning_rate*dw
            b += -learning_rate*db
            loss_list.append(loss)

            if i % 300 == 0:
                print('epoch %d loss %f'%(i, loss))
            
            params = {
                'w':w,
                'b':b
            }
            grads = {
                'dw':dw,
                'db':db
            }
        return loss, loss_list,params, grads
    
    def predict(self, X, params):
        w = params['w']
        b = params['b']
        y_pred = np.dot(X,w)+b
        return y_pred

lasso = Lasso()
loss, loss_list, params, greds = lasso.lasso_train(X_train,y_train, 0.01, 3000)
print(params)
y_pred = lasso.predict(X_test, params)
print(r2_score(y_test, y_pred))

## sklearn 实现
from sklearn.linear_model import Lasso
sk_lasso = Lasso(alpha = 0.01)
sk_lasso.fit(X_train,y_train)
print('sklearn Lasso Intercept:',sk_lasso.intercept_)
print('sklearn Lasso coefficients:', sk_lasso.coef_)
print('sklearn Lasso number of interations:',sk_lasso.n_iter_)

##########################################################################
# Ridge
class Ridge():
    def __init__(self):
        pass

    def initialize_params(self, dims):
        """初始化参数"""
        w = np.zeros((dims,1))
        b = 0
        return w,b
    
    def sign(self, x):
        """定义符号函数并进行向量化,用于对L1正则化项的梯度计算"""
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0
    
    def l2_norm(self,X,y,w,b,alpha):
        """定义Ridge损失函数"""
        num_train = X.shape[0]
        num_feature = y.shape[1]
        y_hat = np.dot(X,w)+b
        loss = np.sum((y_hat-y)**2)+alpha*(np.sum(np.square(w)))# Loss计算与Lasso不同
        dw = np.dot(X.T,(y_hat-y))/num_train+2*alpha*w # db 计算相应也不同
        db = np.sum((y-y_hat))/num_train
        return y_hat, loss, dw, db
    
    def ridge_train(self,X, y, learning_rate=0.01, epochs=1000):
        """定义 Ridge 训练函数"""
        loss_list = []
        w, b = self.initialize_params(X.shape[1])
        for i in range(1,epochs):
            y_hat, loss, dw, db = self.l2_norm(X,y,w,b,0.1) # 改成l2计算
            w += -learning_rate*dw
            b += -learning_rate*db
            loss_list.append(loss)

            if i % 100 == 0:
                print('epoch %d loss %f'%(i, loss))
            
            params = {
                'w':w,
                'b':b
            }
            grads = {
                'dw':dw,
                'db':db
            }
        return loss, loss_list,params, grads
    
    def predict(self, X, params):
        w = params['w']
        b = params['b']
        y_pred = np.dot(X,w)+b
        return y_pred

ridge = Ridge()
loss, loss_list, params, greds = Ridge.ridge_train(X_train,y_train, 0.01, 1000)
print(params)
y_pred = lasso.predict(X_test, params)
print(r2_score(y_test, y_pred))
