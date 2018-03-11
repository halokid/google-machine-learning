#coding=utf-8

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format


#加载数据集
california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")


california_housing_dataframe = california_housing_dataframe.reindex(
  #随机化排序处理
  np.random.permutation(california_housing_dataframe.index)
)
#这一列除以 1000， 就是以千位单位
california_housing_dataframe["median_house_value"] /= 1000.0

#检查数据, 展示一次数据的描述， 以确保数据是正确的
# print (california_housing_dataframe.describe())


'''
构建第一个模型， 尝试预测 median_house_value (房屋价值)， 以 total_rooms (房间的总数) 作为输入特征
'''

#第一步， 定义特征并配置特征列

#声明输入的特征：  total_rooms
#注意 这里定义特征是两个 [], 所以特征的话，有可能是  list的输入方法来的， 因为可以有多个特征
my_feature = california_housing_dataframe[["total_rooms"]]
#为 total_rooms 配置特征列
feature_columns = [tf.feature_column.numeric_column("total_rooms")]

#注意：total_rooms 数据的形状是一维数组（每个街区的房间总数列表）。这是 numeric_column 的默认形状，因此我们不必将其作为参数传递。



#第二步： 定义目标
#定义标签（也就是目标）
#注意这里定义 标签 竟然是 一个 []
targets = california_housing_dataframe["median_house_value"]



#第三步： 配置 LinearRegressor， 配置这个是作为 线性回归模型的， 我们要用特征数据来训练这个模型，
#而小批量随机梯度下降法（SGD），这些其实是训练模型的其中一种算法来的
#注意：为了安全起见，我们还会通过 clip_gradients_by_norm 将梯度裁剪应用到我们的优化器。梯度裁剪可确保梯度大小在训练期间不会变得过大，梯度过大会导致梯度下降法失败。
'''
这里要仔细理解 梯度 和  速率 两个概念

首先从数学角度来看， 梯度是一个 方向和大小的矢量， 也就是说 是一个描述方向和跨度大小 的这个一个东西， 这个方向可以是二维， 也可以是三维方向, 跨度就更好理解了~这是一个值

速率的话， 是学习速率（有时也成为步长）的标量， 可以理解为 是 多少倍 的 梯度， 比如我们的 速率 是 0.01， 假如 梯度的大小为 2.5， 那么梯度下降法就会选择 距离前一个点的
0.01 * 2.5 = 0.025 的位置作为下一个点， 记得这个位置的方向是 梯度 那里已经定好了的
'''

# gradient descent 就是 梯度下降翻译过来
my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
#设置  速率 为 5.0
my_optimizer = tf.contrib.estimator.cli_gradients_by_norm(my_optimizer, 5.0)

#用我们的特征 和 算法去配置 线性回归模型
#设置 学习速率为 0.00001 的 随机下降梯度 算法
linear_regressor = tf.estimator.LinearRegressor(
  feature_columns = feature_columns,
  optimizer= my_optimizer
)


































