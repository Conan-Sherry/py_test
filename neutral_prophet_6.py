"""NeuralProphet之六：多元时间序列预测
NeuralProphet 通过滞后回归(Lagged Regressors)为时间序列预测目标加入其他协变量。"""
import pandas as pd
import matplotlib.pyplot as plt
from neuralprophet import NeuralProphet, set_log_level
set_log_level("ERROR")
df_ercot = pd.read_csv("examples/load_ercot_regions.csv")
df_ercot_y = pd.read_csv("examples/load_ercot.csv")
df_ercot['y'] = df_ercot_y['y']
# print(df_ercot.info())


regions = list(df_ercot)[1:-1]

df = df_ercot
m = NeuralProphet(
    n_forecasts=5,
    n_lags=6,
    learning_rate=0.01,
)
"""注意Prophet:使用model.add_regressor
   NeutralProphet:使用model.add_lagged_regressor"""
m = m.add_lagged_regressor(names=regions)
m.highlight_nth_step_ahead_of_each_forecast(m.n_forecasts)
metrics = m.fit(df, freq="H")

forecast = m.predict(df)
print(forecast.columns)
# fig = m.plot(forecast)
fig1 = m.plot(forecast[-365*24:],plotting_backend='matplotlib')
fig2 = m.plot(forecast[-7*24:],plotting_backend='matplotlib')
# comp = m.plot_components(forecast[-7*24:])
param = m.plot_parameters(plotting_backend='matplotlib')

plt.show()