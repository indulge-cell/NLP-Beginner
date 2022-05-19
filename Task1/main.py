import numpy
import csv
import random
from feature import Bow, N_gram
from comparsion_plot import alpha_gradient_plot

# 数据读取
with open('E:/learn/NLP-Beginner/Task1/train1/train.tsv') as f:
    tsvreader = csv.reader(f, delimiter='\t')
    temp = list(tsvreader)

# 初始化
data = temp[1:]
max_item = 1000
random.seed(2021)
numpy.random.seed(2021)

# 特征提取
bag = Bow(data, max_item)
bag.get_words()
bag.get_matrix()

gram = N_gram(data, dimension=2, max_item=max_item)
gram.get_words()
gram.get_matrix()

# 画图
alpha_gradient_plot(bag, gram, 10000, 10)  # 计算10000次
alpha_gradient_plot(bag, gram, 100000, 10)  # 计算100000次
