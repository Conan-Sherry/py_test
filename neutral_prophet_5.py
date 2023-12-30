# NeuralProphet之五：多时序预测模型
"""实际环境下，我们可能会遇到多个时序数据，比如同一小区的不同楼栋的用电量预测，虽然不同楼栋的用电量幅值差异较大，
但是他们之间的数据周期性还是有些相似的，因此我们需要global model来处理多时序预测。"""
import pandas as pd
import matplotlib.pyplot as plt
from neuralprophet import NeuralProphet, set_log_level
set_log_level("ERROR")
# 以自带的美国的电力消耗数据(load_ercot_regions.csv)为例，首先加载数据
df_ercot = pd.read_csv("examples/load_ercot_regions.csv")
df_dict=pd.read_csv("examples/ercot_load.csv")
# print(df_ercot.head())

#step1:数据集预处理
regions = list(df_ercot)[1:-1]#不取WEST
# print(regions)

df_list = list()
df_dict = {}
for cols in regions:
    aux = df_ercot[['ds', cols]].copy() #select column associated with region
    aux = aux.iloc[:26301, :].copy() #selects data up to 26301 row (2004 to 2007 time stamps)
    aux = aux.rename(columns = {cols: 'y'}) #rename column of data to 'y' which is compatible with Neural Prophet
    df_list.append(aux)
    df_dict[cols] = aux
# df_dict=pd.DataFrame(df_dict)
# print(df_dict)
# 假设已知COAST, EAST两个区域的历史数据，未来也预测这两个区域的数据。
m = NeuralProphet(n_lags=24, normalize='minmax')
"""ValueError: Provided DataFrame (df) must be of pd.DataFrame type.
"""
df_train_dict, df_test_dict = m.split_df(df_dict, valid_p=0.33, local_split=True)
choose_df_train_dict = dict((k, df_train_dict[k]) for k in ['COAST', 'EAST'])
metrics1 = m.fit(choose_df_train_dict, freq = 'H')
# 预测
choose_df_test_dict = dict((k, df_test_dict[k]) for k in ['COAST', 'EAST'])
future = m.make_future_dataframe(choose_df_test_dict, n_historic_predictions=True)
forecast = m.predict(future)
fig1 = m.plot(forecast['COAST'])
fig2 = m.plot(forecast['EAST'])
plt.show()
