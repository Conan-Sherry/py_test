"""ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependen
cy conflicts.
prophet 1.1.4 requires holidays>=0.25, but you have holidays 0.21.13 which is incompatible.
torchaudio 0.7.2 requires torch==1.7.1, but you have torch 1.13.1 which is incompatible.
torchvision 0.8.2+cu101 requires torch==1.7.1, but you have torch 1.13.1 which is incompatible.
"""
import torch
import cv2 as cv
"""
neuralprophet 0.6.2 requires plotly-resampler<0.9.0.0,>=0.8.3.1, but you have plotly-resampler 0.9.1 which is incompatible.
"""
# 之一：安装和使用
import matplotlib.pyplot as plt
from neuralprophet import NeuralProphet
# from neuralprophet.forecaster import _make_future_dataframe
import pandas as pd
# import plotly
# 读入数据集，
# 下面实例中使用的是 佩顿 · 曼宁的维基百科主页 每日访问量的时间序列数据（2007/12/10 - 2016/01/20）
df = pd.read_csv('examples/example_wp_log_peyton_manning.csv')
# print(df.head())

# 调用
m = NeuralProphet(epochs=50)

metrics=m.fit(df)
df_future=m.make_future_dataframe(df,periods=365)
forecast = m.predict(df)
forecast2=m.predict(df_future)
forcast3=pd.concat([forecast,forecast2])

# 誤差
# error=(forecast['y']-forecast['yhat1'])/forecast['y']
# plt.plot(error)
# print(forecast)#2007-12-10-----2016-01-20
# print(forecast2)#2016-01-21

# 可视化结果：
# fig_forecast1 = m.plot(forecast,plotting_backend="matplotlib")
# fig_forecast2 = m.plot(forecast2,plotting_backend="matplotlib")
fig_forecast3=m.plot(forcast3,plotting_backend="matplotlib")

fig_components = m.plot_components(forecast,plotting_backend="matplotlib")
fig_model = m.plot_parameters(plotting_backend="matplotlib")
# fig_forecast1.show()
# fig_forecast2.show()
# fig_components.show()
# fig_model.show()
plt.show()

# 预测
# m = NeuralProphet(epochs=50).fit(df, freq="D")
# df_future = m.make_future_dataframe(df,periods=30)
# forecast = m.predict(df_future)
# fig_forecast = m.plot(forecast)
# fig_forecast.show()

