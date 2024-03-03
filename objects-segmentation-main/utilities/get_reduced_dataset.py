import pandas as pd
import json
import os
import numpy as np
with open('../config.json', 'r') as f:
    config = json.load(f)

csv_path = os.path.join(
    config['Dataset']['paths']['path_dataset'],
    config['Dataset']['paths']['path_csv']
)
print("csv_path: ", csv_path)
df = pd.read_csv(csv_path)


print(df.head())


def unique_count(df):   
    print("len df: ", len(df))    
    category_counts = df["Category"].value_counts()
    category_percentages = df["Category"].value_counts(normalize=True)
    print("category_counts: ", category_counts)
    print("category_percentages: ", np.round(category_percentages, 2))

unique_count(df)

n_smaller = len(df)*0.1
df_smaller = df.sample(n=int(n_smaller))

unique_count(df_smaller)

'''
df_smaller.to_csv(os.path.join(
    config['Dataset']['paths']['path_dataset'],
    "Train_Test_smaller.csv"
), index=False)
'''