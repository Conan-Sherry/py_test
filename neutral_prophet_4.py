#NeuralProphet之四：事件(Events).在预测问题中，经常需要考虑反复出现的特殊事件。

import matplotlib.pyplot as plt
from neuralprophet import NeuralProphet
import pandas as pd
from matplotlib import rcParams
rcParams['font.family'] = 'SimHei'
df = pd.read_csv('examples/example_wp_log_peyton_manning.csv')
# print(df.head())

# 首先生成事件训练数据
playoffs_history = pd.DataFrame({
    'event': 'playoff',
    'ds': pd.to_datetime(['2008-01-13', '2009-01-03', '2010-01-16',
                          '2010-01-24', '2010-02-07', '2011-01-08',
                          '2013-01-12', '2014-01-12', '2014-01-19',
                          '2014-02-02', '2015-01-11', '2016-01-17']),
})
superbowls_history = pd.DataFrame({
    'event': 'superbowl',
    'ds': pd.to_datetime(['2010-02-07', '2014-02-02']),
})
history_events_df = pd.concat((playoffs_history, superbowls_history))
print(history_events_df)


# 生成事件预测数据
playoffs_future = pd.DataFrame({
    'event': 'playoff',
    'ds': pd.to_datetime(['2016-01-21', '2016-02-07'])
})

superbowl_future = pd.DataFrame({
    'event': 'superbowl',
    'ds': pd.to_datetime(['2016-01-23', '2016-02-07'])
})

future_events_df = pd.concat((playoffs_future, superbowl_future))
print(future_events_df)


#通过add_events函数为NeuralProphet对象添加事件配置
m = NeuralProphet(epochs=50,
    n_forecasts=10,
    n_lags=12,
    seasonality_mode="multiplicative",
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=True,
)
m = m.add_events(["superbowl", "playoff"],mode="multiplicative",regularization=0.02)

history_df = m.create_df_with_events(df, history_events_df)
print(history_df)

metrics = m.fit(history_df, freq="D")

forecast = m.predict(df=history_df)
fig = m.plot(forecast,plotting_backend='matplotlib')
plt.title("peyton_manning(佩顿 · 曼宁的维基百科主页 每日访问量)")
plt.show()
