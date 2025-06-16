import math
from collections import Counter
import pandas as pd
from io import StringIO

csv_data = """ID,Feature1,Feature2,Label
1,6,2,Class A
2,2,3,Class A
3,3,1,Class B
4,5,5,Class B
"""

df = pd.read_csv(StringIO(csv_data))

features = df.columns[1:-1]  
label_col = df.columns[-1]   

def get_neighbors(df, query, k):
    distances = []
    for i, row in df.iterrows():
        row_features = row[features]
        dist = math.sqrt(sum((row_features[j] - query[j]) ** 2 for j in range(len(features))))
        distances.append((dist, i))
    distances.sort(key=lambda x: x[0])
    return distances[:k]

def get_prediction(df, neighbors):
    labels = [df.iloc[i][label_col] for _, i in neighbors]
    return Counter(labels).most_common(1)[0][0]

query = [float(input(f"Enter {col}: ")) for col in features]
k = int(input("Enter value of k: "))

neighbors = get_neighbors(df, query, k)

print("\nNearest Neighbors:")
for dist, idx in neighbors:
    print(f"Index: {idx}, Distance: {dist:.2f}, Label: {df.iloc[idx][label_col]}")

prediction = get_prediction(df, neighbors)
print(f"\nPredicted Class: {prediction}")
