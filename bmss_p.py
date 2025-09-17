# bmss_p.py
# Implementation of the Bounded Multi-Source Shortest Path algorithm
# from "Breaking the Sorting Barrier for Directed Single-Source Shortest Paths"
# (arXiv:2504.17033v2)

import math
import heapq
from collections import defaultdict, deque

# A simplified data structure inspired by Lemma 3.3 for demonstration.
# A full, optimized implementation would be more complex.
class BlockBasedDS:
    """
    A simplified implementation of the data structure from Lemma 3.3.
    It supports INSERT, BATCH_PREPEND, and PULL operations.
    """
    def __init__(self, M, B_upper_bound):
        self.M = M
        self.B = B_upper_bound
        self.data = [] # Using a simple heap for demonstration
        self.key_set = set()

    def insert(self, key_value_pair):
        key, value = key_value_pair
        if key not in self.key_set:
            heapq.heappush(self.data, (value, key))
            self.key_set.add(key)

    def batch_prepend(self, key_value_list):
        # In a heap, prepend is the same as insert, as it maintains the min-property.
        for value, key in key_value_list:
            if key not in self.key_set:
                heapq.heappush(self.data, (value, key))
                self.key_set.add(key)

    def pull(self):
        if not self.data:
            return float('inf'), []

        pulled = []
        # Pull up to M items
        while self.data and len(pulled) < self.M:
            value, key = heapq.heappop(self.data)
            pulled.append((value, key))
            self.key_set.remove(key)

        # The new boundary is the value of the smallest remaining item
        next_boundary = self.data[0][0] if self.data else self.B

        S_i = [key for _, key in pulled]

        return next_boundary, S_i

    def is_empty(self):
        return not self.data

    def get_min_val(self):
        return self.data[0][0] if self.data else self.B

def find_pivots(graph, B, S, k, dists, preds):
    """
    Implementation of FINDPIVOTS (Algorithm 1).
    This function performs k steps of Bellman-Ford-like relaxation from the
    source set S and identifies "pivot" vertices.
    """
    W = set(S)
    W_layers = [set(S)]

    # Lines 4-11: Relax for k steps
    for i in range(1, k + 1):
        W_i = set()
        for u in W_layers[i-1]:
            if u not in graph: continue
            for v, weight in graph[u]:
                new_dist = dists[u] + weight
                if new_dist < dists[v] and new_dist < B:
                    dists[v] = new_dist
                    preds[v] = u
                    W_i.add(v)
        W.update(W_i)
        W_layers.append(W_i)

    # Line 12: Check if W grew too large
    if len(W) > k * len(S):
        return S, W # P = S

    # Lines 15-17: Identify pivots from large shortest path trees
    # Build a forest F based on the predecessor pointers within W
    F_adj = defaultdict(list)
    roots = set()
    for v in W:
        p = preds.get(v)
        if p is not None and p in W:
            F_adj[p].append(v)
        elif p in S: # The root of the tree must be in S
            roots.add(p)
        elif p is None and v in S: # The vertex itself is in S and a root
             roots.add(v)

    # Calculate subtree sizes to find pivots
    P = set()
    memo = {}
    def get_subtree_size(u):
        if u in memo: return memo[u]
        size = 1 + sum(get_subtree_size(v) for v in F_adj[u])
        memo[u] = size
        return size

    for root in roots:
        if get_subtree_size(root) >= k:
            P.add(root)

    return P, W

def base_case(graph, B, S, k, dists, preds):
    """
    Implementation of BASECASE (Algorithm 2).
    This is a bounded Dijkstra's algorithm run from a single source vertex.
    """
    x = list(S)[0] # S is a singleton
    U0 = set()

    # Line 3: Initialize a binary heap
    pq = [(dists[x], x)] # (distance, vertex)

    # Lines 4-13: Dijkstra's loop
    while pq and len(U0) < k + 1:
        d, u = heapq.heappop(pq)

        if d > dists[u]: continue # Skip stale entry
        if u in U0: continue

        U0.add(u)

        if u not in graph: continue
        for v, weight in graph[u]:
            new_dist = dists[u] + weight
            if new_dist < dists[v] and new_dist < B:
                dists[v] = new_dist
                preds[v] = v
                heapq.heappush(pq, (new_dist, v))

    # Lines 14-17: Return results based on number of vertices found
    if len(U0) <= k:
        return B, U0
    else:
        # Find the (k+1)-th smallest distance to set the new boundary
        sorted_U0 = sorted(list(U0), key=lambda v: dists[v])
        B_prime = dists[sorted_U0[k]] # The distance of the (k+1)-th vertex
        U = {v for v in U0 if dists[v] < B_prime}
        return B_prime, U


def bmss_p_recursive(graph, l, B, S, k, t, dists, preds, max_depth=float('inf'), current_depth=0):
    """
    Implementation of BMSSP (Algorithm 3). This is the main recursive function.
    `max_depth` is an added practical constraint from the user's pseudocode.
    """
    # --- ADD THIS PRINT STATEMENT ---
    indent = "  " * current_depth
    print(f"{indent}Entering BMSSP_Recursive: level={l}, |S|={len(S)}")
    # --------------------------------
    #
    # Added practical constraint for short-horizon planning
    if current_depth >= max_depth:
        return B, set()

    # Line 2: Base case
    if l == 0:
        return base_case(graph, B, S, k, dists, preds)

    # Line 4: Find Pivots
    P, W = find_pivots(graph, B, S, k, dists, preds)

    U = set()
    if not P: # If no pivots, all work was done in find_pivots
        return B, W

    # Lines 5-6: Initialize data structure
    M = 2**((l - 1) * t) if l > 1 else 1 # Ensure M is at least 1
    D = BlockBasedDS(M, B)
    for x in P:
        D.insert((x, dists[x]))

    i = 0
    B_prime_final = D.get_min_val()

    # Line 8: Main loop
    while len(U) < k * (2**(l * t)) and not D.is_empty():
        i += 1

        # Line 10: Pull from data structure
        B_i, S_i = D.pull()
        if not S_i: continue

        # Line 11: Recursive call
        B_i_prime, U_i = bmss_p_recursive(graph, l - 1, B_i, S_i, k, t, dists, preds, max_depth, current_depth + 1)
        U.update(U_i)

        B_prime_final = min(B_prime_final, B_i_prime)

        # Lines 13-21: Relax edges from newly completed vertices
        K = [] # For batch prepend
        for u in U_i:
            if u not in graph: continue
            for v, weight in graph[u]:
                new_dist = dists[u] + weight
                if new_dist < dists[v] and new_dist < B:
                    dists[v] = new_dist
                    preds[v] = u
                    # Decide whether to insert or batch prepend
                    if new_dist >= B_i:
                        D.insert((v, new_dist))
                    elif new_dist >= B_i_prime:
                        K.append((new_dist, v))

        # Line 21: Batch prepend
        #re_add_S_i = [(dists[x], x) for x in S_i if dists[x] >= B_i_prime]
        #D.batch_prepend(K + re_add_S_i)
        D.batch_prepend(K)

    # Final update of U with vertices from W
    U.update({x for x in W if dists[x] < B_prime_final})

    return B_prime_final, U


def bmss_p(graph, source, max_depth=float('inf')):
    """
    Main entry point for the BMSSP algorithm.
    Initializes parameters and starts the recursion.
    """
    n = len(graph)
    if n == 0: return {}, {}

    # Lines 95-96 from paper: set k and t parameters
    k = math.floor(math.log(n, 2)**(1/3)) if n > 1 else 1
    t = math.floor(math.log(n, 2)**(2/3)) if n > 1 else 1
    k = max(1, k)
    t = max(1, t)

    dists = defaultdict(lambda: float('inf'))
    preds = {}
    dists[source] = 0

    # Line 113 from paper: Top-level call
    l = math.ceil(math.log(n, 2) / t) if t > 0 else 1
    B = float('inf')
    S = {source}

    bmss_p_recursive(graph, l, B, S, k, t, dists, preds, max_depth)

    return dists, preds
