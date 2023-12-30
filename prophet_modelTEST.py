import cv2 as cv
import json
from prophet.serialize import model_from_json
import matplotlib.pyplot as plt
# 导入模型
with open("peyton_manning_model.json","r") as fin:
    peyton_manning_model=model_from_json(json.load(fin))


# 构建待预测日期数据框，periods = 365 代表除历史数据的日期外再往后推 365 天
future = peyton_manning_model.make_future_dataframe(periods=365)

print(future.tail())


# 预测数据集
forecast = peyton_manning_model.predict(future)
peyton_manning_end=forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
print(peyton_manning_end)
peyton_manning_end.to_csv("peyton_manning_prediction.csv")

# 展示预测结果
fig1=peyton_manning_model.plot(forecast);

# 预测的成分分析绘图，展示预测中的趋势、周效应和年度效应
fig2=peyton_manning_model.plot_components(forecast);



fig1.show()
