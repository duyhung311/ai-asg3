import random
import sys, csv
from collections import Counter
from math import inf
from typing import List


STUDENT_ID = 'a1903971' # your student ID
DEGREE = 'PG' # or PG if you are in the postgraduate course

class KdNode:
    def __init__(self, point, d, val):
        self.point = point
        self.d = d
        self.val = val
        self.left = None
        self.right = None

def generate_sample_indexes(N: int, N_prime: int, n_trees: int, rand_seed: int) -> List[int]:
    index_list = [i for i in range(0, N)]  # create a list of indexes for all data
    sample_indexes = []
    for j in range(0, n_trees):
        random.seed(rand_seed + j)  # random_seed is one of the input parameters
        subsample_idx = random.sample(index_list, k=N_prime)  # create unique N’ indices
        sample_indexes = sample_indexes + subsample_idx
    return sample_indexes

star_dim = 0  # Global variable to capture starting dimension
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


def KdForest(data, d_list, rand_seed):
    forest = []
    n_trees = len(d_list)
    N = len(data)
    N_prime = round(N * 0.8)  # N' is 80% of N

    # Get the sample indexes
    sample_indexes = generate_sample_indexes(N, N_prime, n_trees, rand_seed)
    count = 0
    for i in range(n_trees):
        # Sequentially get N' indexes for this tree
        start = i * N_prime
        end = start + N_prime
        indexes = sample_indexes[start:end]

        sampled_data = [data[j] for j in indexes]  # data[j] is a (x, y) pair

        # Build the k-d tree with the specified depth
        global start_dim
        start_dim = d_list[i]  
        tree = build_kd_tree(sampled_data, d_list[i])
        forest.append(tree)

    return forest

def euclidean_dist2(p1, p2):
    dist = 0
    for i in range(10):
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
        
    go_left_or_right = target[d] <= node.val

    first = node.left if go_left_or_right else node.right
    second = node.right if go_left_or_right else node.left

    best, best_dist = find_1nn(first, target, best, best_dist)

    if abs(target[d] - node.val) < best_dist:
        best, best_dist = find_1nn(second, target, best, best_dist)

    return best, best_dist

def find_most_frequent_data_point(dps):
    # Convert arrays to tuples (hashable) for counting
    tuple_arrays = [tuple(arr) for arr in dps]
    
    # Count occurrences
    counter = Counter(tuple_arrays)
    
    # Get the most common
    most_common_tuple, count = counter.most_common(1)[0]
    
    return {
        'array': list(most_common_tuple),
        'count': count
    }

def Predict_KdForest(forest, x):

    labels = []
    for tree in forest:
        label, _ = find_1nn(tree, x)  # Nearest neighbor search
        labels.append(label)
        # print(f"label: {label}")
        # label_dict[label] = dist3
    # Majority vote
    most_common = find_most_frequent_data_point(labels)['array']
    return most_common

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

train_file = sys.argv[1]
test_file = sys.argv[2]
rand_seed = int(sys.argv[3])
n_trees_str = sys.argv[4]
numbers = n_trees_str.strip('[]').split(',')
n_trees = [int(x.strip()) for x in numbers]

train_data = read_data(train_file)
test_data = read_data(test_file)

# tree = build_kd_tree(train_data, D=start_dim)
forest = KdForest(train_data, n_trees, rand_seed=rand_seed)

for i, test_point in enumerate(test_data):
    nn = Predict_KdForest(forest, test_point)
    print(f"{int(nn[11])}")
# python nn_kdforest.py train test-sample 42 [1,2,3]
