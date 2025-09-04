import sys
import pandas as pd
import numpy as np

data  = pd.read_csv(r"F:\BaiduNetdiskDownload\PM2.5\train.csv",encoding="big5")

data = data.iloc[:,3:]
#在 Pandas 中，表达式 data[data == 'NR'] = 0的索引部分 data == 'NR'是一个 ​​布尔索引（Boolean Indexing）​​ 操作，
# 它的核心逻辑是生成一个与 data形状相同的布尔值矩阵（True/False），标记所有值为 'NR'的位置为 True，其他为 False。
data[data == 'NR'] = 0
raw_data = data.to_numpy()

data = {}
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[:, day * 24 : (day + 1) * 24] = raw_data[18 * (20 * month + day) : 18 * (20 * month + day + 1), :]
    data[month] = sample

#这里是提取出特征和标签，其中不够的地方略去

x = np.empty([12*471,18*9],dtype=float)
y = np.empty([12*471,1],dtype=float)

for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day==19 and hour > 14:
                break
            x[471*month+24*day+hour,:]= data[month][:,24*day+hour:24*day+hour+9].reshape(1,-1)
            y[471*month+24*day+hour,0] = data[month][9,24*day+hour+9]

#按列归一化处理，你需要记住np.mean np.std

mean_x = np.mean(x,axis=0) #9*18
std_x = np.std(x,axis=0)

for i in range(12*471):
    for j in range(18*9):
        if std_x[j]!=0 :
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]

#将数据集分为训练和测试

import math
x_train_set = x[:math.floor(len(x)*0.8),:]
y_train_set = y[:math.floor(len(y)*0.8),:]
x_test_set = x[math.floor(len(x)*0.8):,:]
y_test_set = y[math.floor(len(y)*0.8):,:]


dim = 9*18+1
##w就是初始化随机值了
w = np.empty([dim,1])
x = np.concatenate((x,np.ones([dim,1])),axis=1)
iter_times = 1000
###ada优化
learning_rate = 100
adagrad = np.zeros([dim,1])
eps = 0.00000001

for i in range(iter_times):
    ##损失我们最好计算一下，这样直观显示我们的学习的是否收敛
    ##我们可以计算平均平方根 损失
    loss = np.sqrt(np.sum(np.power(y-np.dot(x,w),2))/471/12)
    gradient = np.dot(2*x.transpose,np.dot(x,w)-y)
    adagrad += gradient ** 2
    w = w - learning_rate*gradient / np.sqrt(adagrad+eps)
np.save('',w)


