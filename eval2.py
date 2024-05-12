# import pandas as pd

# # Read CSV files into pandas dataframes
# train = pd.read_csv('data/train_latlon.csv')
# test = pd.read_csv('data/test_latlon.csv')

# # Concatenate both dataframes vertically
# dataset = pd.concat([train, test], ignore_index=True)

# # Save the combined dataframe to a CSV file
# dataset.to_csv('data/dataset.csv', index=False)

# import pandas as pd

# # Read the CSV file
# df = pd.read_csv('data/dataset.csv')

# # Drop unwanted columns
# df.drop(['label', 'tid', 'day', 'hour', 'category'], axis=1, inplace=True)

# # Round to 4 decimals
# df['lat'] = df['lat'].round(3)
# df['lon'] = df['lon'].round(3)

# # Drop duplicates based on lat and lon pairs
# df.drop_duplicates(subset=['lat', 'lon'], keep='first', inplace=True)

# # Reorder columns to have lat and lon as the first two columns
# df = df[['lat', 'lon'] + [col for col in df.columns if col not in ['lat', 'lon']]]

# # Write the modified dataframe back to a new CSV file
# df.to_csv('data/gps1.csv', index=False)

import pandas as pd
hotspot =[]

df1 = pd.read_csv("data/gps1.csv")
df2 = pd.read_csv("results/epsilon2000/syn_traj_test_2000.csv")


df2['lat'] = df2['lat'].round(3)
df2['lon'] = df2['lon'].round(3)


# Number of matching pairs between the two tables
matching_pairs = df2.merge(df1[['lat', 'lon']], on=['lat', 'lon'], how='inner')

# # Calculate the percentage
percentage = (len(matching_pairs) / len(df2)) * 100

print(f"Retention percentage: {percentage:.10f}%")


import pandas as pd
from shapely.geometry import LineString
import numpy as np


df3 = pd.read_csv('data/test_latlon.csv')

# Initialize dictionaries to store LineStrings for each tid from both files
line_dict1 = {}
line_dict2 = {}

# Calculating centroid from LineString
def calculate_centroid(line_coords):
    line = LineString(line_coords)
    centroid = line.centroid
    return (centroid.x, centroid.y)

# Function to process data from DataFrame and populate line_dict
def process_data(df, line_dict):
    for tid, group in df.groupby('tid'):
        line_coords = [(row['lon'], row['lat']) for index, row in group.iterrows()]
        line_dict[tid] = line_coords


process_data(df2, line_dict1)
process_data(df3, line_dict2)

# Calculate centroids for each tid from both files
centroids1 = np.array([calculate_centroid(line_dict1[tid]) for tid in line_dict1])
centroids2 = np.array([calculate_centroid(line_dict2[tid]) for tid in line_dict2])

# Calculate the difference between centroids using manhattan distance
manhattan_distance = np.mean(np.sum(np.abs(centroids2 - centroids1), axis=1))

print("Manhattan distance:")
print(manhattan_distance)



