# NeuralProphet之七：NeuralProphet + Optuna
"""ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
neuralprophet 0.6.2 requires plotly<6.0.0,>=5.13.1, but you have plotly 5.10.0 which is incompatible.
pydantic 2.3.0 requires typing-extensions>=4.6.1, but you have typing-extensions 4.5.0 which is incompatible.
pydantic-core 2.6.3 requires typing-extensions!=4.7.0,>=4.6.0, but you have typing-extensions 4.5.0 which is incompatible.
torchaudio 0.7.2 requires torch==1.7.1, but you have torch 1.13.1 which is incompatible.
torchvision 0.8.2+cu101 requires torch==1.7.1, but you have torch 1.13.1 which is incompatible.
"""
from copy import deepcopy
import optuna
import pandas as pd
import matplotlib.pyplot as plt
from neuralprophet import NeuralProphet, set_log_level
set_log_level("ERROR")


# 导入数据
df_train = pd.read_csv("examples/shenghuoqu0815.csv")
df_test = pd.read_csv("examples/shenghuoqu0821.csv")
# print(df_train.info())
# print(df_test.info())

# 1 Baseline
n_lags = 12
n_forecasts=11
# n_lags = n_lags,n_forecasts= n_forecasts,
# changepoints_range=0.85, n_changepoints=20,
"""没有optuna"""
# m = NeuralProphet(epochs=50,n_lags = n_lags,n_forecasts= n_forecasts,
#                   yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=True,
#                   normalize='minmax')
# # m.add_seasonality()
# metrics = m.fit(df_train, freq='5min')
# forecast_train = m.predict(df_train)
# forecast_test = m.predict(df_test)
# print(forecast_train.columns)
# fig = m.plot(forecast_train,plotting_backend="matplotlib")
# fig = m.plot(forecast_test,plotting_backend="matplotlib")
# fig_param = m.plot_parameters(plotting_backend="matplotlib")
# plt.show()

# m.add_country_holidays()
"""预测优化"""
# 2 加入Optuna,定义Optuna参数的搜索范围
default_para = dict(n_lags = n_lags,n_forecasts= n_forecasts,changepoints_range=0.8,n_changepoints=5, trend_reg=0.1,normalize='minmax',learning_rate =1)

param_types = dict(changepoints_range='float',n_changepoints='int',trend_reg='float',learning_rate='float',month_order = 'int',week_order ='int')
bounds = {'changepoints_range': [0.6,0.8,0.9],
          'n_changepoints': [4, 8],
          'trend_reg': [0.001, 1],
          'learning_rate': [0.001, 1],
          'day_order': [1, 7],
          }

def create_nph(**para):
    temp_para = deepcopy(para)
    day_order = temp_para.pop('day_order')
    m = NeuralProphet(**temp_para)
    m = m.add_seasonality('my_day', 24, day_order)
    return m


def nph_warper(trial,ts):
    params = {}
    params['changepoints_range'] = trial.suggest_categorical('changepoints_range', bounds['changepoints_range'])
    params['n_changepoints'] = trial.suggest_int('n_changepoints', bounds['n_changepoints'][0], bounds['n_changepoints'][1])
    params['trend_reg'] = trial.suggest_float('trend_reg', bounds['trend_reg'][0], bounds['trend_reg'][1])
    params['learning_rate'] = trial.suggest_float('learning_rate', bounds['learning_rate'][0], bounds['learning_rate'][1])
    params['day_order'] = trial.suggest_int('day_order', bounds['day_order'][0], bounds['day_order'][1])
    temp_para = deepcopy(default_para)
    temp_para.update(params)
    # METRICS = ['SmoothL1Loss', 'MAE', 'RMSE']
    METRICS=['Loss_test', 'MAE_val', 'RMSE_val']
    metrics_test = pd.DataFrame(columns=METRICS)
    m = create_nph(**temp_para)
    folds = m.crossvalidation_split_df(ts, freq="H", k=5, fold_pct=0.2, fold_overlap_pct=0.5)
    for df_train, df_test in folds:
        m = create_nph(**temp_para)
        train = m.fit(df_train)
        test = m.test(df=df_test)
        metrics_test = metrics_test.append(test[METRICS].iloc[-1])
        # metrics_test = pd.concat(metrics_test,test[METRICS].iloc[-1])
    out = metrics_test['MAE_val'].mean()
    return out


def objective(trial):
    ts = df_train.copy() #select column associated with region
    return nph_warper(trial,ts)

# Optuna 调优
study = optuna.create_study(direction='minimize',study_name='electricity_analysis',
            load_if_exists=True, storage="sqlite:///electricity_consumption.db")
study.optimize(objective, n_trials=3)
print(study.best_trial)
print(study.best_params)


# 根据最优参数构建NeuralProphet
best_para = deepcopy(default_para)
best_para.update(study.best_params)
m = create_nph(**best_para)

# 训练和预测
metrics = m.fit(df_train, freq='5min')
forecast_train = m.predict(df_train)
forecast_test = m.predict(df_test)

fig = m.plot(forecast_train,plotting_backend="matplotlib")
fig = m.plot(forecast_test,plotting_backend="matplotlib")
fig_param = m.plot_parameters(plotting_backend="matplotlib")
plt.show()


