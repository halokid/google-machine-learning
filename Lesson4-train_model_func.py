#coding=utf-8

"""
这个是包含 “调整模型超参数”  统一函数这个版本的， 这个函数 其实包括了   Lesson4.py  源码里面的几个流程，包含如下：

包含了 第三，五， 六 步
"""

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




def train_model(learning_rate, steps, batch_size, input_feature="total_rooms"):
  """Trains a linear regression model of one feature.

  Args:
    learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    input_feature: A `string` specifying a column from `california_housing_dataframe`
      to use as input feature.
  """
  periods = 10
  steps_per_period = steps / periods

  my_feature = input_feature
  my_feature_data = california_housing_dataframe[[my_feature]]
  my_label = "median_house_value"
  targets = california_housing_dataframe[my_label]

  # Create feature columns
  feature_columns = [tf.feature_column.numeric_column(my_feature)]

  # Create input functions
  training_input_fn = lambda:my_input_fn(my_feature_data, targets, batch_size=batch_size)
  prediction_input_fn = lambda: my_input_fn(my_feature_data, targets, num_epochs=1, shuffle=False)

  # Create a linear regressor object.
  my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  linear_regressor = tf.estimator.LinearRegressor(
    feature_columns=feature_columns,
    optimizer=my_optimizer
  )

  # Set up to plot the state of our model's line each period.
  plt.figure(figsize=(15, 6))
  plt.subplot(1, 2, 1)
  plt.title("Learned Line by Period")
  plt.ylabel(my_label)
  plt.xlabel(my_feature)
  sample = california_housing_dataframe.sample(n=300)
  plt.scatter(sample[my_feature], sample[my_label])
  colors = [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]

  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print ("Training model...")
  print ("RMSE (on training data):")
  root_mean_squared_errors = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    linear_regressor.train(
      input_fn=training_input_fn,
      steps=steps_per_period
    )
    # Take a break and compute predictions.
    predictions = linear_regressor.predict(input_fn=prediction_input_fn)
    predictions = np.array([item['predictions'][0] for item in predictions])

    # Compute loss.
    root_mean_squared_error = math.sqrt(
      metrics.mean_squared_error(predictions, targets))
    # Occasionally print the current loss.
    print ("  period %02d : %0.2f" % (period, root_mean_squared_error))
    # Add the loss metrics from this period to our list.
    root_mean_squared_errors.append(root_mean_squared_error)
    # Finally, track the weights and biases over time.
    # Apply some math to ensure that the data and line are plotted neatly.
    y_extents = np.array([0, sample[my_label].max()])

    weight = linear_regressor.get_variable_value('linear/linear_model/%s/weights' % input_feature)[0]
    bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

    x_extents = (y_extents - bias) / weight
    x_extents = np.maximum(np.minimum(x_extents,
                                      sample[my_feature].max()),
                           sample[my_feature].min())
    y_extents = weight * x_extents + bias
    plt.plot(x_extents, y_extents, color=colors[period])
  print ("Model training finished.")

  # Output a graph of loss metrics over periods.
  plt.subplot(1, 2, 2)
  plt.ylabel('RMSE')
  plt.xlabel('Periods')
  plt.title("Root Mean Squared Error vs. Periods")
  plt.tight_layout()
  plt.plot(root_mean_squared_errors)

  # Output a table with calibration data.
  calibration_data = pd.DataFrame()
  calibration_data["predictions"] = pd.Series(predictions)
  calibration_data["targets"] = pd.Series(targets)
  display.display(calibration_data.describe())

  print ("Final RMSE (on training data): %0.2f" % root_mean_squared_error)



# train_model(learning_rate=0.00001, steps=100, batch_size=1)
# train_model(learning_rate=0.001, steps=100, batch_size=1)
#fixme: batch_size  的意思就是  每次传输 多大的数据给 训练模型？？？  比如一列的话， 意思是指每次传送 这列 里面多少行数据 给模型吗？？？
train_model(learning_rate=0.00002, steps=500, batch_size=5)

#fixme: 调参的一些经验建议
"""
有适用于模型调整的标准启发法吗？
这是一个常见的问题。简短的答案是，不同超参数的效果取决于数据。因此，不存在必须遵循的规则，您需要对自己的数据进行测试。

即便如此，我们仍在下面列出了几条可为您提供指导的经验法则：

训练误差应该稳步减小，刚开始是急剧减小，最终应随着训练收敛达到平稳状态。
如果训练尚未收敛，尝试运行更长的时间。
如果训练误差减小速度过慢，则提高学习速率也许有助于加快其减小速度。
但有时如果学习速率过高，训练误差的减小速度反而会变慢。
如果训练误差变化很大，尝试降低学习速率。
较低的学习速率和较大的步数/较大的批量大小通常是不错的组合。
批量大小过小也会导致不稳定情况。不妨先尝试 100 或 1000 等较大的值，然后逐渐减小值的大小，直到出现性能降低的情况。
重申一下，切勿严格遵循这些经验法则，因为效果取决于数据。请始终进行试验和验证。
"""















