from collections import Counter
import math

data = [
    ([6, 2], 'Class A'),
    ([2, 3], 'Class A'),
    ([3, 1], 'Class B'),
    ([5, 5], 'Class B')
]

def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def knn(test_point, data, k=3):
    distances = []
    for features, label in data:
        dist = euclidean_distance(test_point, features)
        distances.append((dist, label))
    distances.sort(key=lambda x: x[0])  

    k_labels = [label for _, label in distances[:k]]

    most_common = Counter(k_labels).most_common(1)[0][0]
    return most_common


test_point = [2, 1]
predicted_class = knn(test_point, data, k=3)
print(f"The test point {test_point} is classified as: {predicted_class}")
