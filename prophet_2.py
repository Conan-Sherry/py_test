# 2.1 预测饱和增长
import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet
df = pd.read_csv('examples/example_wp_log_R.csv')
df['cap'] = 8.5

m = Prophet(growth='logistic')

m.fit(df)

future = m.make_future_dataframe(periods=1826)
future['cap'] = 8.5
fcst = m.predict(future)
fig = m.plot(fcst)
plt.show()