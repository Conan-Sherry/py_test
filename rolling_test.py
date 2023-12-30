import pandas as pd
s=pd.Series([1,2,3,4,5])
x=s.rolling(3,min_periods=2).sum()
print(x)