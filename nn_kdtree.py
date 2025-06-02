import sys
import csv
import math

STUDENT_ID = 'a1903971' # your student ID
DEGREE = 'PG' # or PG if you are in the postgraduate course

class KdNode:
    def __init__(self, point=None, d=None, val=None):
        self.point = point  # The actual point (only at the split)
        self.d = d          # Split dimension
        self.val = val      # Median value on that dimension
        self.left = None
        self.right = None

def build_kd_tree(P, D=0):
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
        median_point = P_sorted[idx2]
        median_val = (P_sorted[idx1][d] + P_sorted[idx2][d]) / 2

    # Prevent infinite loop if all values at d are the same
    if all(p[d] == median_val for p in P):
        return KdNode(point=median_point, d=d, val=median_val)

    node = KdNode(point=median_point, d=d, val=median_val)

    # "Yellow" version — left <= median, right > median
    left_points = [p for p in P_sorted if p[d] <= median_val and p != median_point]
    right_points = [p for p in P_sorted if p[d] > median_val]

    node.left = build_kd_tree(left_points, D + 1)
    node.right = build_kd_tree(right_points, D + 1)

    return node


def euclidean_dist2(p1, p2):
    return sum((a - b) ** 2 for a, b in zip(p1, p2))

    
# def nearest_neighbor(kd_tree, point, D=0):

def find_1nn(root, query_point, depth=0, best=None, best_dist=float('inf')):
    if root is None:
        return best, best_dist

    d = root.d
    axis = d % 11

    # Distance to current node
    dist = euclidean_dist2(query_point, root.point)
    if dist < best_dist:
        best = root.point
        best_dist = dist

    # Choose side to search first
    go_left = query_point[axis] < root.val
    first = root.left if go_left else root.right
    second = root.right if go_left else root.left

    # Search the good side first
    best, best_dist = find_1nn(first, query_point, depth + 1, best, best_dist)

    # Check if we need to search the other side
    if abs(query_point[axis] - root.val) ** 2 < best_dist:
        best, best_dist = find_1nn(second, query_point, depth + 1, best, best_dist)

    return best, best_dist
    
    
# ---------- CSV Loader ----------
def load_data(path):
    # with open(path, newline='') as f:
    #     reader = csv.reader(f)
    #     data = [[float(x) for x in row] for row in reader]

    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]
        data = [[float(x) for x in line.split()] for line in lines[1:]]
    print(f"Loaded {len(data)} points from {path}")
    if not data:
        print("No data found in the file.")
        return []
    return data

# ---------- Main ----------
print("Running nn_kdtree.py...")
print(len(sys.argv))
if len(sys.argv) != 4:
    print("Usage: python nn_kdtree.py [train] [test] [dimension]")
    sys.exit(1)
else:
    print(f"Student ID: {STUDENT_ID}, Degree: {DEGREE}")

train_file = sys.argv[1]
test_file = sys.argv[2]
start_dim = int(sys.argv[3])

train_data = load_data(train_file)
test_data = load_data(test_file)

tree = build_kd_tree(train_data, D=start_dim)

for i, test_point in enumerate(test_data):
    nn, dist2 = find_1nn(tree, test_point)
    print(f"Test point {i + 1}:")
    print(f"Nearest Neighbor: {nn[11]}")
    print(f"Distance: {math.sqrt(dist2):.4f}")
    print()


# python nn_kdtree.py train test-sample 3
