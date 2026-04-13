import pandas as pd

df = pd.read_excel("data/Problem_C_Data_Wordle.xlsx", header=1)

# print(df.shape)
# print(df.columns.tolist())
# print(df.head())
# print(df.dtypes)

row = df.iloc[0]
cols = ['1 try','2 tries','3 tries','4 tries','5 tries','6 tries','7 or more tries (X)']
print(row[cols].tolist())
print("sum:", row[cols].sum())