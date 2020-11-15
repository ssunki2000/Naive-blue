import numpy as np
import matplotlib.pyplot as plt


class BPNN:
    def __init__(self, nn_shape=[2, 4, 1]):
        self.W = []                                                                              # 权重
        self.B = []                                                                              # 阈值
        self.O = []                                                                              # 各神经元节点输出
        self.grads = []                                                                          # bp算法中误差与神经节点输入的微分(梯度项)

        self.mean = np.zeros(nn_shape[2])
        self.mean = self.mean.reshape((1, nn_shape[2]))

        self.W_shape = []                                                                       
        self.B_shape = []
        self.O_shape = []
        self.grads_shape = []

        self.errs = []                                                                           # 记录每次迭代的误差误差

        for index in range(len(nn_shape) - 1):                                                   # 初始化W,B,O,grads矩阵
            self.W.append(2 * np.random.random([nn_shape[index], nn_shape[index + 1]]) - 1)
            self.W[index] = self.W[index].reshape([nn_shape[index], nn_shape[index + 1]])
            self.W_shape.append(self.W[index].shape)

            self.B.append(2 * np.random.random(nn_shape[index + 1]) - 1)
            self.B[index] = self.B[index].reshape(1, nn_shape[index + 1])
            self.B_shape.append(self.B[index].shape)

            self.O.append(np.zeros(nn_shape[index + 1]))
            self.O[index] = self.O[index].reshape(1, nn_shape[index + 1])
            self.O_shape.append(self.O[index].shape)

            self.grads.append(np.zeros(nn_shape[index + 1]))
            self.grads[index] = self.grads[index].reshape(1, nn_shape[index + 1])
            self.grads_shape.append(self.grads[index].shape)

        self.y_hat = self.O[-1]
        self.y_hat = self.y_hat.reshape(self.O[-1].shape)

        print('建立{}层神经网络网络'.format(len(nn_shape)))
        print(self.W_shape)
        print(self.B_shape)
        print(self.O_shape)
        print(self.grads_shape)

    def sigmoid(self, x):
       
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_derivate(self, x):
       
        return x * (1 - x)

    def error(self, y, y_hat):
        err = y - y_hat
        return 0.5 * err.dot(err.T)

    def cross_entropy(self, y, y_hat):
        tmp = np.argwhere(y == 1)
        return -np.log(y_hat[0, tmp[0, 1]])

    def softmax(self, x):
        exp_all = np.exp(x)
        return exp_all / np.sum(exp_all)

    def update_output(self, x, x_istest=False):
        '''
        更新各神经元的输出值，x为n*1向量
        '''
        if x_istest == True:
            x = (x - self.mean) / self.var

        for index in range(len(self.O)):
            if index == 0:
                self.O[index] = self.sigmoid(
                    x.dot(self.W[index]) + self.B[index])
            elif index == len(self.O) - 1:
                self.O[index] = self.softmax(
                    self.O[index - 1].dot(self.W[index]) + self.B[index])
            else:
                self.O[index] = self.sigmoid(
                    self.O[index - 1].dot(self.W[index]) + self.B[index])

            self.O[index] = self.O[index].reshape(self.O_shape[index])

        self.y_hat = self.O[-1]
        self.y_hat = self.y_hat.reshape(self.O[-1].shape)
        return self.y_hat

    def update_grads(self, y):
        '''
        更新梯度值，y为p*1向量
        '''
        for index in range(len(self.grads) - 1, -1, -1):
            if index == len(self.grads) - 1:
                '''#该代码用来计算使用均方误差和sigmoid函数的二分类问题
                self.grads[index] = self.sigmoid_derivate(
                    self.O[index]) * (y - self.O[index])
                '''
                tmp = np.argwhere(y == 1)

                for index_g in range(self.grads[index].shape[1]):
                    if index_g == tmp[0, 1]:
                        self.grads[index][0, index_g] = 1 - self.O[index][0, index_g]
                    else:
                        self.grads[index][0, index_g] = - self.O[index][0, index_g]
            else:                                                                             # 链式法则计算隐含层梯度
                self.grads[index] = self.sigmoid_derivate(
                    self.O[index]) * self.W[index + 1].dot(self.grads[index + 1].T).T

            self.grads[index] = self.grads[index].reshape(
                self.grads_shape[index])

    def update_WB(self, x, learning_rate):
        for index in range(len(self.W)):
            if index == 0:

                self.W[index] += learning_rate * x.T.dot(self.grads[index])
                self.B[index] -= learning_rate * self.grads[index]
            else:
                self.W[index] += learning_rate * self.O[index - 1].T.dot(self.grads[index])
                self.B[index] -= learning_rate * self.grads[index]
            self.B[index] = self.B[index].reshape(self.B_shape[index])

    def preprocess(self, X, method='centring'):
        self.mean = np.mean(X, axis=0)
        self.var = X.var()
        X = (X - self.mean) / self.var
        if method == 'centring':
            return X

    def fit(self, X, Y, Preprocess=True, method='centring', thre=0.03, learning_rate=0.001, max_iter=1000):
        '''
        将样本和label输入，X,Y中的样本均为行向量
        '''
        if Preprocess == True:
            X = self.preprocess(X, method=method)

        err = np.inf
        count = 0

        while err > thre:
            err = 0
            for index in range(X.shape[0]):

                x = X[index, :].reshape((1, -1))
                y = Y[index, :].reshape((1, -1))

                self.update_output(x)
                x = X[index, :].reshape((1, -1))
                self.update_grads(y)
                self.update_WB(x, learning_rate=learning_rate)
                err += self.cross_entropy(y, self.y_hat)
            err /= index + 1
            self.errs.append(err)
            count += 1
            if count > max_iter:
                print("超过最大迭代次数{}".format(max_iter))
                break

            print(count)
            print(err)

    def one_hot_label(self, Y):
        '''
        将label转化为0001形式，若label有3种，则转化为100,010,001
        这里的label必须从0开始
        '''
        category = list(set(Y[:, 0]))
        Y_ = np.zeros([Y.shape[0], len(category)])

        for index in range(Y.shape[0]):
            Y_[index, Y[index, 0]] = 1

        return Y_

if __name__ == '__main__':

    from sklearn.datasets import load_digits
    digits = load_digits()
    X = digits.data
    Y = digits.target
    X = X.reshape(X.shape)
    Y = Y.reshape(Y.shape[0], 1)
    bp = BPNN([64, 128, 64, 10])                                                                 #建立神经网络对象
    Y = bp.one_hot_label(Y)

    train_data = X[:1000, :]
    train_label = Y[:1000, :]

    test_data = X[1000:-1, :]
    test_label = Y[1000:-1, :]

    bp.fit(train_data, train_label, Preprocess=True,thre=0.01, learning_rate=0.005, max_iter=1000)#构建网络
    count = 0
    for i in range(test_data.shape[0]):
        x = test_data[i].reshape(1, 64)
        pre = bp.update_output(x, x_istest=True)
        y = test_label[i].reshape(1, 10)
        a = np.where(pre == np.max(pre))
        b = np.where(y == np.max(y))
        if a[1][0] == b[1][0]:
            count += 1


    print('准确率：{}'.format(count / test_label.shape[0]))
    plt.plot(bp.errs)
    plt.show()