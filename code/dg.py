import pandas as pd
df1 = pd.DataFrame({'A': [3,4,6], 'B': [7,0,9]})
df1.to_csv('df1.csv')
df2 = pd.DataFrame({'A':[0,8,3], 'B':[0,3,4]})
df2.to_csv('df2.csv')

df1 = pd.read_csv('df1.csv')
df2 = pd.read_csv('df2.csv')
df = pd.concat([df1,df2], axis=0)
print(df)


