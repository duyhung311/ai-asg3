import sys
import csv
from collections import namedtuple
from math import inf
import math


STUDENT_ID = 'a1903971' # your student ID
DEGREE = 'PG' # or PG if you are in the postgraduate course

# Define KD Node
class KdNode:
    def __init__(self, point, d, val):
        self.point = point
        self.d = d
        self.val = val
        self.left = None
        self.right = None

def read_data(file_path):
    data = []
    with open(file_path, newline='') as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip the header row
        for row in reader:
            try:
                r = row[0].strip().split()
                data.append([float(n) for n in r])
                # data.append(float(n) for n in r)
            except ValueError as e:
                print(f"Skipping row due to conversion error: {row} - {e}")

    return data

# Global variable to capture starting dimension
start_dim = 0

def build_kd_tree(P, D=0):
    global start_dim
    if not P:
        return None
    elif len(P) == 1:
        d = D % 11
        return KdNode(point=P[0], d=d, val=P[0][d])

    d = D % 11
    P_sorted = sorted(P, key=lambda x: x[d])
    n = len(P_sorted)

    if n % 2 == 1:
        median_index = n // 2
        median_point = P_sorted[median_index]
        median_val = median_point[d]
    else:
        idx1 = n // 2
        idx2 = n // 2 - 1
        median_point = P_sorted[idx1]
        median_val = (P_sorted[idx1][d] + P_sorted[idx2][d]) / 2

    if all(p[d] == median_val for p in P):
        return KdNode(point=median_point, d=d, val=median_val)

    node = KdNode(point=median_point, d=d, val=median_val)

    left_points = [p for p in P_sorted if p[d] <= median_val and p != median_point]
    right_points = [p for p in P_sorted if p[d] > median_val]
    if D == start_dim:
        indent = '.' * D
        print(f"{indent}l{len(left_points)}")
        print(f"{indent}r{len(right_points)}")

    node.left = build_kd_tree(left_points, D + 1)
    node.right = build_kd_tree(right_points, D + 1)

    return node

def euclidean_dist2(p1, p2):
    dist = 0
    print(f"p1: {p1}, p2: {p2}")
    for i in range(11):
        diff = p1[i] - p2[i]
        dist += diff * diff
    return dist

def find_1nn(node, target, best=None, best_dist=inf):
    if node is None:
        return best, best_dist

    d = node.d
    point = node.point
    dist_to_point = euclidean_dist2(point[:-1], target)

    if dist_to_point < best_dist:
        best = point
        best_dist = dist_to_point
    go_left = target[d] <= node.val

    first = node.left if go_left else node.right
    second = node.right if go_left else node.left

    best, best_dist = find_1nn(first, target, best, best_dist)

    if abs(target[d] - node.val) < best_dist:
        best, best_dist = find_1nn(second, target, best, best_dist)

    return best, best_dist


train_file = sys.argv[1]
test_file = sys.argv[2]
start_dim = int(sys.argv[3])
# left_or_right = str(sys.argv[4])

train_data = read_data(train_file)
test_data = read_data(test_file)

tree = build_kd_tree(train_data, D=start_dim)

for i, test_point in enumerate(test_data):
    nn, dist2 = find_1nn(tree, test_point)
    print(f"{int(nn[11])}")
    
#python nn_kdtree.py train test-sample 0