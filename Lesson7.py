#!coding=utf-8
'''
把数据分为三部分， 训练， 测试， 验证数据， 用测试数据反复的去 训练 数据模型， 在最后才用 验证数据来验证
'''
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

california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")


def preprocess_features(california_housing_dataframe):
 '''
 输入的数据：
  加利福利亚的 房屋数据， 以 pandas dataframe 的形式输入

 输出的数据：
  一个包含了 特征数据的  dataframe 格式的数据， 以供数据模型去训练
 '''
 selected_features = california_housing_dataframe[
   ["latitude",
    "longitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income"]]

 processed_features = selected_features.copy()  #深拷贝
 #创建一个合成的特征数据
 processed_features["romms_per_person"] = (
   california_housing_dataframe["total_rooms"] / california_housing_dataframe["population"]
 )

 return processed_features



def preprocess_targets(california_housing_dataframe):
  output_targets = pd.DataFrame()
  output_targets["median_house_value"] = (
    california_housing_dataframe["median_house_value"] / 1000.0
  )
  return output_targets



#---------------------------- 下面 是 测试数据 +  训练数据， 统称为 训练集 ------------------------
#从 17000 之中选出前  12000 来做训练数据
training_examples = preprocess_features(california_housing_dataframe.head(12000))
training_examples.describe()


training_targets = preprocess_targets(california_housing_dataframe.head(12000))
training_targets.describe()





#------------------------- 下面是 验证集 数据 ---------------------------------------
#注意验证集 数据里面 也要有 特征数据的 ~~ 最后的验证就是把 验证集 里面的 特征数据 输入给训练模型， 然后再匹配 我们验证集中的标签数据
#以此来做验证， 这样准确率就会增加
validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
validation_examples.describe()

validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))
validation_targets.describe()









