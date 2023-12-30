import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import time
#显示中文
from matplotlib import rcParams
rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus']=False
# rcParams['toolbar'] = 'toolmanager'
actual_values=np.array([300,350,400,450,500])
predicted_values=np.array([280,365,390,430,520])
error=actual_values-predicted_values

df = pd.read_csv('D:\LWF_ProgramProjects\python_test\py_test\examples\example_wp_log_R.csv')
plt.figure(1)
plt.plot(df['y'])
plt.show()
#绘制误差曲线，方法1，简单粗暴
# plt.figure(2)
# plt.plot(error,color='black',alpha=0.8)
# plt.fill_between(range(5),0,error,facecolor='gray',alpha=0.3)
# plt.axhline(y=0,linestyle='--',color='black')
# # rotation=0,loc='center'
# plt.ylabel("实际值-预测值",)
# plt.xlabel("时间")

#绘制误差曲线，方法2，简单粗暴
# fig=plt.figure(2)
# plt.scatter(actual_values,predicted_values,label='实际值 vs 预测值',color='blue')
#
# #
# plt.errorbar(actual_values,predicted_values,
#              linestyle='',color='red',alpha=0.5)
# fig2=plt.plot(actual_values,actual_values,label='理想状态',linestyle='--',color='green')
# plt.legend()
# plt.xlabel('实际值')
# plt.ylabel('预测值')
# fig.canvas.mpl_connect('button_press_event',on_press)
# plt.show()



# 生成数据
# x=np.linspace(0,2*np.pi,100)
# y=np.sin(x)
# # 绘制原始曲线
# # plt.plot(x,y,'-r',label='原始曲线')
# # 创建阴影区域
# plt.figure(3)
# plt.fill_between(x,0,y,facecolor='green',alpha=0.3,interpolate=True,label='阴影区域')
# # 添加图例
# plt.legend(loc='upper left')
# # 显示图形
# plt.show()

# while(1):
#     print('hello')
#     time.sleep(5)

