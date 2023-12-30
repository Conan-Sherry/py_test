# 之二：季节性
import pandas as pd
import matplotlib.pyplot as plt
from neuralprophet import NeuralProphet, set_log_level,save
import datetime
set_log_level("ERROR")

# data_location = "https://raw.githubusercontent.com/ourownstory/neuralprophet-data/main/datasets/"

df = pd.read_csv("examples/example_air_passengers.csv")
# 默认seasonality_mode="additive"预测
# m = NeuralProphet(epochs=50,seasonality_mode="multiplicative")
# m.fit(df, freq="MS")
#
#
# future = m.make_future_dataframe(df, periods=50, n_historic_predictions=len(df))#区别于prophet用法，需要指定
# forecast = m.predict(future)
# fig = m.plot(forecast,plotting_backend="matplotlib")
# plt.show()
# 模型保存
# save(m,"air_passengers_model.json")
# print(forecast.columns)
# print(forecast)


# 模型读取
from neuralprophet import load
m = load("air_passengers_model.json")
# future = m.make_future_dataframe(df, periods=50, n_historic_predictions=len(df))#区别于prophet用法，需要指定
start_time=datetime.datetime(1963,1,1)
end_time=datetime.datetime(1965,2,1)
x_datetime=pd.date_range(start=start_time,end=end_time,freq='M')
future=pd.DataFrame({'ds':x_datetime,'y':None})
print(future)
forecast = m.predict(future)
print(forecast)
fig = m.plot(forecast,plotting_backend="matplotlib")
plt.show()
