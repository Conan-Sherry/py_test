import pandas as pd
from neuralprophet import NeuralProphet, set_log_level
s=pd.Series([1,2,3,4,5])
x=s.rolling(3,min_periods=2).sum()
y=s.rolling(3,min_periods=2).mean()
print(x)
print(y)


m=NeuralProphet()
df3 = pd.DataFrame({'ds': pd.date_range(start = '2022-12-03', periods = 10, freq = 'D'),
                    'y': [7.67, 7.64, 7.55, 8.25, 8.32, 9.59, 8.52, 7.55, 8.25, 8.09]})
folds = m.crossvalidation_split_df(df3, k = 2, fold_pct = 0.2)
print(folds)


