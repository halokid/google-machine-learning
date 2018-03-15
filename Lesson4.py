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
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

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

#fixme: 第一步， 定义特征并配置特征列

#声明输入的特征：  total_rooms
#注意 这里定义特征是两个 [], 所以特征的话，有可能是  list的输入方法来的， 因为可以有多个特征
my_feature = california_housing_dataframe[["total_rooms"]]
#为 total_rooms 配置特征列
feature_columns = [tf.feature_column.numeric_column("total_rooms")]

#注意：total_rooms 数据的形状是一维数组（每个街区的房间总数列表）。这是 numeric_column 的默认形状，因此我们不必将其作为参数传递。



#fixme: 第二步： 定义目标
#定义标签（也就是目标）
#注意这里定义 标签 竟然是 一个 []
targets = california_housing_dataframe["median_house_value"]



#fixme: 第三步： 配置模型， 这个是采用线性回归的模型的， 所以也叫 LinearRegressor，还需要用特征数据来训练这个模型，
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
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

#用我们的特征 和 算法去配置 线性回归模型
#设置 学习速率为 0.00001 的 随机下降梯度 算法
linear_regressor = tf.estimator.LinearRegressor(
  feature_columns = feature_columns,
  optimizer= my_optimizer
)


#fixme: 第四步： 定义输入函数
'''
要将加利福尼亚州住房数据导入 LinearRegressor，我们需要定义一个输入函数，让它告诉 TensorFlow 如何对数据进行预处理，以及在模型训练期间如何批处理、随机处理和重复数据。

首先，我们将 Pandas 特征数据转换成 NumPy 数组字典。然后，我们可以使用 TensorFlow Dataset API 根据我们的数据构建 Dataset 对象，并将数据拆分成大小为 batch_size 的多批数据，以按照指定周期数 (num_epochs) 进行重复。

注意：如果将默认值 num_epochs=None 传递到 repeat()，输入数据会无限期重复。

然后，如果 shuffle 设置为 True，则我们会对数据进行随机处理，以便数据在训练期间以随机方式传递到模型。buffer_size 参数会指定 shuffle 将从中随机抽样的数据集的大小。

最后，输入函数会为该数据集构建一个迭代器，并向 LinearRegressor 返回下一批数据。
'''
def my_input_fn(features, targets, batch_size = 1, shuffle = True, num_epochs = None):
  #转换pandas读取到的内容为 一个 字典，  以 np arrays 的方式
  features = {key:np.array(value) for key, value in dict(features).items()}

  #构造数据， 然后配置 批数据处理， 比如重复次数
  ds = Dataset.from_tensor_slices((features, targets))    #警告， 2GB的大小限制
  ds = ds.batch(batch_size).repeat(num_epochs)

  #重新打乱数据， 如果指定了要打乱的话
  if shuffle:
    ds = ds.shuffle(buffer_size=10000)

  #返回下一个批处理的数据
  features, labels = ds.make_one_shot_iterator().get_next()
  return features, labels




#fixme: 第五步：  训练模型， 也就是训练第三步定义的模型
#首先我们训练 100 步
_ = linear_regressor.train(
  input_fn = lambda: my_input_fn(my_feature, targets),
  steps = 100
)




#fixme: 第六步：  评估模型
'''
我们基于该训练数据做一次预测，看看我们的模型在训练期间与这些数据的拟合情况。

注意：训练误差可以衡量您的模型与训练数据的拟合情况，但并不能衡量模型泛化到新数据的效果。在后面的练习中，您将探索如何拆分数据以评估模型的泛化能力。
'''
#为预测数据创建一个输入函数
#注意： 当每一次单独的预测开始计算的时候， 我们在当次计算中不需要 重复 打乱数据
prediction_input_fn = lambda: my_input_fn(my_feature, targets, num_epochs=1, shuffle=False)

#调用 predict()函数 实现 线性回归 算法， 去进行预测
predictions = linear_regressor.predict(input_fn=prediction_input_fn)

#fixme: 格式化预测数据 以  numpy array 的数据形式， 以便我们可以 计算 错误的度量
predictions = np.array([item['predictions'][0] for item in predictions])

#fixme: 这里不明白， 输出  均方误差(MSE),  predictions 就是我们预测的结果，  而 targets 就是我们的 目标结果 ， 两个进行误差的对比， 看看我们预测得准不准确
mean_squared_error = metrics.mean_squared_error(predictions, targets)
#fixme: 这里不明白啊，把 均方误差的值 用来 开平台根, 这个是L2的算法？？
root_mean_squared_error = math.sqrt(mean_squared_error)

print ("Mean Squared Error (on training data):    %0.3f"  % mean_squared_error)
print ("Root Mean Squared Error (on training data):      %0.3f" % root_mean_squared_error)


'''
这是出色的模型吗？您如何判断误差有多大？

由于均方误差 (MSE) 很难解读，因此我们经常查看的是均方根误差 (RMSE)。RMSE 的一个很好的特性是，它可以在与原目标相同的规模下解读。

我们来比较一下 RMSE 与目标最大值和最小值的差值：
'''
min_house_value = california_housing_dataframe["median_house_value"].min()
max_house_value = california_housing_dataframe["median_house_value"].max()
min_max_difference = max_house_value - min_house_value

print ("Min. Median House Value:     %0.3f" % min_house_value)
print ("Max. Median House Value:     %0.3f" % max_house_value)

print ("Difference between Min. and Max.:        %0.3f" % min_max_difference)
print ("Root Mean Squared Error:              %0.3f" % root_mean_squared_error)

'''
下面就是上面的代码数据的结果：

Mean Squared Error (on training data):    56367.025
Root Mean Squared Error (on training data):      237.417
Min. Median House Value:     14.999
Max. Median House Value:     500.001
Difference between Min. and Max.:        485.002
Root Mean Squared Error:              237.417

---------------------------------------------------------------

其实 这个结果已经告诉我们，这样的预测方法是误差是多大， 因为 我们预测的数据，其实已经有的了， 只是 机器学习 会学习一部分数据， 然后预测到的结果，去跟
我们现在已经有的正确数据进行对比， 而 这个 对比的标准就是 RMSE 这个标准, 我们可以看到 这个标准很大， 因为我们的房价 最大值是 500.001， 最小值是 14.999,
而我们的误差标准 去到了    237.417 ， 这个差不多是  最大房价 的一半了， 所以效果不理想
'''


'''
我们的误差跨越目标值的近一半范围，可以进一步缩小误差吗？

这是每个模型开发者都会烦恼的问题。我们来制定一些基本策略，以降低模型误差。

首先，我们可以了解一下根据总体摘要统计信息，预测和目标的符合情况。
'''

'''
下面这里是 进行 predictions 和  target 的对比， 按照这个代码~~  预测的结果非常非常垃圾...........
'''
calibration_data = pd.DataFrame()
calibration_data["predictions"] = pd.Series(predictions)
calibration_data["target"] = pd.Series(targets)
print (calibration_data.describe())



#------------------------------------------ 下面开始画图展示一下 -----------------------------------------------
sample = california_housing_dataframe.sample(n = 300)

#获取最小 和 最大 的  total_rooms 的值
x_0 = sample["total_rooms"].min()
x_1 = sample["total_rooms"].max()

#检索在训练期间最后的权重 和 偏差
weight = linear_regressor.get_variable_value("linear/linear_model/total_rooms/weights")[0]
bias = linear_regressor.get_variable_value("linear/linear_model/bias_weights")

#获得 (根据 min  和 max  的  total_rooms_values 的值来 预测 median_house_values 的值)
y_0 = weight * x_0 + bias
y_1 = weight * x_1 + bias

#描绘回归 线形 从 (x_0, y_0) 到  (x_1, y_1)












