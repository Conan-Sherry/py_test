import matplotlib.pyplot as plt
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice
import logging
import sys
import optuna
import matplotlib.pyplot as plt
def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x-2)**2

# 通过调用create_study()来创建持久性研究，研究使用SQLite文件进行自动记录
# add stream handler of stdout to show the messages
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "example-study"
storage_name = "sqlite:///{}.db".format(study_name)
study = optuna.create_study(study_name=study_name, storage=storage_name,load_if_exists=True)
study.optimize(objective, n_trials=10)

df = study.trials_dataframe(attrs=("number",  "value", "params", "state"))
# print(df)


plot_optimization_history(study).show() # 绘制优化历史 (重要)

# plot_intermediate_values(study) # 绘制学习曲线（重要）
# plot_parallel_coordinate(study) # 绘制高维参数曲线
# # plot_parallel_coordinate(study, params=['bagging_freq','bagging_fraction']) # 选择绘制的参数
# plot_contour(study) # 绘制参数之间的关系图（重要）
# # plot_contour(study, params=['bagging_freq','bagging_fraction']) # 选择绘制的参数
# plot_slice(study) # 绘制参数的分面图，显示单个参数的调参过程
# # plot_slice(study, params=['bagging_freq','bagging_fraction']) # 选择绘制的参数
# plot_param_importances(study) # 绘制参数重要性图（重要）
# plot_edf(study) # 绘制经验分布曲线
# plt.show()
