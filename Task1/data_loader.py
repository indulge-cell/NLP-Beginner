'''
 bag of words
'''
import pandas as pd
from nltk.tokenize import word_tokenize
import numpy as np
from collections import defaultdict
from nltk.corpus import stopwords
import time

def data_loader(fileName):
    # 加载数据
    data = pd.read_csv(fileName, sep='\t', header=0)
    return data

def word_extraction(sentence):
    # 提取句子中的词,去掉部分停用词
    words = sentence.split()
    stop_word = set(stopwords.words('english'))
    cleaned_text = [w.lower() for w in words if not w in stop_word]
    return cleaned_text

def tokenize(data):
    # 生产词表
    words = []
    for sentence in data:
        w = word_extraction(sentence)
        words.extend(w)
    words = sorted(list(set(words)))
    return words

def bow(data):
    vec_matrix = []
    vocab = tokenize(data)
    for sentence in data:
        words = word_extraction(sentence)
        vector = np.zeros(len(vocab))
        for w in words:
            for i, word in enumerate(vocab):
                if word == w:
                    vector[i] += 1
        vec_matrix.append(vector)
    return vec_matrix

def sigmoid(x):
    y = 1.0/(1.0 + np.exp(-x))
    return y

def loss_function(theta, x_b, y):
    p_predict = sigmoid(x_b.dot(theta))
    try:
        return -np.sum(y*np.log(p_predict) + (1-y)*np.log(1-p_predict))
    except:
        return float('inf')

def d_loss_func(theta, x_b, y):
    out = sigmoid(x_b.dot(theta))
    return x_b.T.dot(out-y) / len(x_b)

def gradient_descent(x_b, y, theta0, eta, n_iters=1e4, epsilon=1e-8):
    iter = 0
    while iter < n_iters:
        gradient = d_loss_func(theta0, x_b, y)
        theta1 = theta0
        theta0 = theta0 - eta * gradient
        iter += 1
        if abs(loss_function(theta0, x_b, y)-loss_function(theta1, x_b, y)) <epsilon:
            break
    return theta0

def fit(train_data, train_label, eta=0.01, n_iters=1e4):
    assert train_data.shape[0] == train_label.shape[0],'训练数据集的长度需要和标签长度保持一致'
    x_b = np.hstack([np.ones((train_data.shape[0], 1)), train_data])
    theta0 = np.zeros(x_b.shape[1])
    theta0 = gradient_descent(x_b, train_label, theta0, eta, n_iters)
    # intercept = theta0[0]
    # coef = theta0[1:]
    return theta0

def predict_proba(x_predict):
    x_b = np.hstack([np.ones((len(x_predict)), 1), x_predict])

def predict(x_predict):
    proba = predict_proba(x_predict)
    return np.array(proba > 0.5, dtype='int')

def loadDataSet(x,y):
    y = np.mat(y)
    b = np.ones(y.shape)
    X = np.column_stack((b, x))
    X = np.mat(X)
    label_type = np.unique(y.tolist())
    eyes = np.eye(len(label_type))
    Y = np.zeros((X.shape[0], len(label_type)))
    for i in range(X.shape[0]):
        Y[i, :] = eyes[int(y[i, 0])]
    return X, y, Y

def data_convert(x, y):
    b = np.ones(y.shape)
    x_b = np.column_stack((b, x))
    k = len(np.unique(y.tolist()))
    eyes_mat = np.eye(k)
    y_onehot = np.zeros((y.shape[0], k))
    for i in range(0, y.shape[0]):
        y_onehot[i] = eyes_mat[y[i]]
    return x_b, y, y_onehot

def softmax(s):
    return np.exp(s) / np.sum(np.exp(s), axis=1)

def loss_function1(theta, x, y):
    p_predict = sigmoid(x.dot(theta))
    try:
        return -np.sum(y*np.log(p_predict) + (1-y)*np.log(1-p_predict))
    except:
        return float('inf')



def gradAscent(x, y, eta=0.05, n_iters=500):
    m = np.shape(x)[1]  # x的特征数
    n = np.shape(y)[1]  # y的分类数
    weights = np.ones((m, n))  # 权重矩阵
    for k in range(n_iters):
        h = softmax(x * weights)
        error = y - h
        weights = weights + eta * x.transpose() * error  # 梯度下降算法公式
    return weights.getA()

def SoftmaxSGD(x, y, eta=0.05, n_iters=50):
    # 随机梯度上升算法
    m = np.shape(x)[1]
    n = np.shape(y)[1]
    weights = np.ones((m, n))

    for k in range(n_iters):
        for i in range(0, len(x)):
            h = softmax(x[i] * weights)
            error = y[i] - h[0]
            weights = weights + eta * x[i].T * error[0]  # 随机梯度下降算法公式
    return weights.getA()

def softmax_predict(weights, testdata):
    y_hat = softmax(testdata*weights)
    predicted = y_hat.argmax(axis=1).getA()
    return predicted

if __name__ == '__main__':
    begin = time.time()
    fileName1 = 'E:/learn/NLP-Beginner/Task1/train1/train.tsv'
    fileName2 = 'E:/learn/NLP-Beginner/Task1/test1/test.tsv'
    train = data_loader(fileName1)
    print(train.head())
    print(train['Sentiment'].unique())
    test = data_loader(fileName2)
    print(test.head())
    i = test['Phrase'][0:2]
    result = bow(i)
    print(result)
    print(len(result), len(result[0]))
    end = time.time()
    print('Running time: %s Seconds' % (end - begin))







