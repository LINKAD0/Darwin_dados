import numpy as np
import pandas as pd


# path_csv = 'output/active_learning/query_image_names_preselected.csv'
path_csv = 'output/active_learning/query_image_names_all.csv'

df = pd.read_csv(path_csv)

# print(df)
# df.groupby('team')['points'].nunique()
print("Number of unique platforms:",df['platform'].nunique())
print("Occurences of each individual platform:",df['platform'].value_counts())
