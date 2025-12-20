import pandas as pd
df = pd.read_csv('d:/Ozone_Project_7th_dec/final_cal.csv', nrows=0)
with open('cols.txt', 'w') as f:
    for col in df.columns:
        f.write(col + '\n')
        # print(col)

# print(df.head())