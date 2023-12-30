# import baostock as bs
import matplotlib.pyplot as plt
import pandas as pd
from neuralprophet import NeuralProphet
'''step1:获取数据集'''
# #### 登陆系统 ####
# lg = bs.login()
# # 显示登陆返回信息
# print('login respond error_code:' + lg.error_code)
# print('login respond  error_msg:' + lg.error_msg)
#
# #### 获取沪深A股历史K线数据 ####
# # 详细指标参数，参见“历史行情指标参数”章节；“分钟线”参数与“日线”参数不同。“分钟线”不包含指数。
# # 分钟线指标：date,time,code,open,high,low,close,volume,amount,adjustflag
# # 周月线指标：date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg
#
# rs = bs.query_history_k_data_plus("sh.000001",
#                                   "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
#                                   start_date='2008-05-30', end_date='2023-9-28',
#                                   frequency="d")
# print('query_history_k_data_plus respond error_code:' + rs.error_code)
# print('query_history_k_data_plus respond  error_msg:' + rs.error_msg)
#
# #### 打印结果集 ####
# data_list = []
# while (rs.error_code == '0') & rs.next():
#     # 获取一条记录，将记录合并在一起
#     data_list.append(rs.get_row_data())
# result = pd.DataFrame(data_list, columns=rs.fields)
#
# #### 结果集输出到csv文件 ####
# result.to_csv("sh.000001.csv", index=False)
# print(result)
#
# #### 登出系统 ####
# bs.logout()

#读取数据
data = pd.read_csv('./sh.000001.csv')
# print(data.head())

prcp_data = data.rename(columns={'date': 'ds', 'close': 'y'})[['ds', 'y']]

""" 
注解：
n_forecasts：表示要预测多少天的数据
n_lags：表示要往前观察多少天的数据
n_changepoints：表示我们认为多少天股市会有一次大的变动
yearly_seasonality：将其设置为True表示我们认为股市每一年会有一个周期性的变化

因此，下面模型的意思是，用过去120天的数据来预测未来30天的上证指数变化，
同时我们认为上证指数180天会与一次大的变化，而一年内会有一次周期性的变动。
"""
model = NeuralProphet(
    n_forecasts=30,
    n_lags=120,
    n_changepoints=180,
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    batch_size=64,
    epochs=3,
    learning_rate=0.3)
model.set_plotting_backend("matplotlib")

metrics = model.fit(prcp_data, freq='D')

future = model.make_future_dataframe(prcp_data, periods=30, n_historic_predictions=True)

forcast = model.predict(future)
# print(forcast)
# forcast.to_csv('shuju.csv')
forecasts_plot = model.plot(forcast)
plt.show()