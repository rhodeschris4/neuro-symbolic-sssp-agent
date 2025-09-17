#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include <limits>
#include <map>
#include <unordered_set>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

namespace py = pybind11;

// Type definitions for clarity
using Edge = std::pair<int, double>; // {neighbor_vertex, weight}
using Graph = std::vector<std::vector<Edge>>;
using Distances = std::vector<double>;
using Predecessors = std::vector<int>;
using VertexSet = std::unordered_set<int>;

// A simplified BlockBasedDS using a C++ priority queue (min-heap)
class BlockBasedDS {
private:
    using PQElement = std::pair<double, int>; // {distance, vertex}
    std::priority_queue<PQElement, std::vector<PQElement>, std::greater<PQElement>> pq;
    std::unordered_set<int> key_set;

public:
    void insert(int key, double value) {
        if (key_set.find(key) == key_set.end()) {
            pq.push({value, key});
            key_set.insert(key);
        }
    }

    void batch_prepend(const std::vector<std::pair<double, int>>& items) {
        for (const auto& item : items) {
            insert(item.second, item.first);
        }
    }

    std::pair<double, std::vector<int>> pull(int M) {
        if (is_empty()) {
            return {std::numeric_limits<double>::infinity(), {}};
        }

        std::vector<int> pulled_keys;
        for (int i = 0; i < M && !pq.empty(); ++i) {
            pulled_keys.push_back(pq.top().second);
            key_set.erase(pq.top().second);
            pq.pop();
        }

        double next_boundary = is_empty() ? std::numeric_limits<double>::infinity() : pq.top().first;
        return {next_boundary, pulled_keys};
    }

    bool is_empty() const {
        return pq.empty();
    }
};

// --- Forward Declarations ---
std::pair<double, VertexSet> bmss_p_recursive(const Graph& graph, int l, double B, const VertexSet& S, int k, int t, Distances& dists, Predecessors& preds, int max_depth, int current_depth, int initial_l);

// Algorithm 2: Base Case (Bounded Dijkstra)
std::pair<double, VertexSet> base_case(const Graph& graph, double B, const VertexSet& S, int k, Distances& dists, Predecessors& preds) {
    int start_node = *S.begin();
    VertexSet U0;

    using PQElement = std::pair<double, int>;
    std::priority_queue<PQElement, std::vector<PQElement>, std::greater<PQElement>> pq;
    pq.push({dists[start_node], start_node});

    while (!pq.empty() && U0.size() < static_cast<size_t>(k + 1)) {
        auto [d, u] = pq.top();
        pq.pop();

        if (d > dists[u] || U0.count(u)) {
            continue;
        }
        U0.insert(u);

        if (static_cast<size_t>(u) < graph.size()) {
            for (const auto& edge : graph[u]) {
                int v = edge.first;
                double weight = edge.second;
                if (dists[u] + weight < dists[v] && dists[u] + weight < B) {
                    dists[v] = dists[u] + weight;
                    preds[v] = u;
                    pq.push({dists[v], v});
                }
            }
        }
    }

    if (U0.size() <= static_cast<size_t>(k)) {
        return {B, U0};
    } else {
        std::vector<int> sorted_U0(U0.begin(), U0.end());
        std::sort(sorted_U0.begin(), sorted_U0.end(), [&](int a, int b) {
            return dists[a] < dists[b];
        });
        double B_prime = dists[sorted_U0[k]];
        VertexSet U;
        for (int v : U0) {
            if (dists[v] < B_prime) {
                U.insert(v);
            }
        }
        return {B_prime, U};
    }
}


// Algorithm 1: Find Pivots
std::pair<VertexSet, VertexSet> find_pivots(const Graph& graph, double B, const VertexSet& S, int k, Distances& dists, Predecessors& preds) {
    VertexSet W = S;
    std::vector<VertexSet> W_layers(k + 1);
    W_layers[0] = S;

    for (int i = 1; i <= k; ++i) {
        for (int u : W_layers[i - 1]) {
            if (static_cast<size_t>(u) < graph.size()) {
                for (const auto& edge : graph[u]) {
                    int v = edge.first;
                    double weight = edge.second;
                    if (dists[u] + weight < dists[v] && dists[u] + weight < B) {
                        dists[v] = dists[u] + weight;
                        preds[v] = u;
                        W_layers[i].insert(v);
                    }
                }
            }
        }
        W.insert(W_layers[i].begin(), W_layers[i].end());
    }

    // The paper's logic for pivot selection is complex.
    // We are using a fallback mentioned in the paper for when W grows large. [cite: 138-139]
    if (W.size() > static_cast<size_t>(k) * S.size()) {
        return {S, W};
    }

    // A full implementation would build a predecessor forest to find large subtrees.
    // For simplicity, we stick to the paper's fallback logic.
    return {S, W};
}


// Algorithm 3: Main Recursive Function
std::pair<double, VertexSet> bmss_p_recursive(const Graph& graph, int l, double B, const VertexSet& S, int k, int t, Distances& dists, Predecessors& preds, int max_depth, int current_depth, int initial_l) {
    if (current_depth >= max_depth) {
        return {B, {}};
    }
    if (l == 0) {
        return base_case(graph, B, S, k, dists, preds);
    }

    auto [P, W] = find_pivots(graph, B, S, k, dists, preds);
    VertexSet U;

    if (P.empty()) {
        return {B, W};
    }

    int M = (l > 1) ? static_cast<int>(pow(2, (l - 1) * t)) : 1;
    BlockBasedDS D;
    double B_prime_final = std::numeric_limits<double>::infinity();

    for (int x : P) {
        D.insert(x, dists[x]);
        if (dists[x] < B_prime_final) {
            B_prime_final = dists[x];
        }
    }

    while (U.size() < static_cast<size_t>(k) * pow(2, l * t) && !D.is_empty()) {
        auto [B_i, S_i_vec] = D.pull(M);
        if (S_i_vec.empty()) continue;

        VertexSet S_i(S_i_vec.begin(), S_i_vec.end());

        auto [B_i_prime, U_i] = bmss_p_recursive(graph, l - 1, B_i, S_i, k, t, dists, preds, max_depth, current_depth + 1, initial_l);
        U.insert(U_i.begin(), U_i.end());
        B_prime_final = std::min(B_prime_final, B_i_prime);

        std::vector<std::pair<double, int>> K;
        for (int u : U_i) {
            if (static_cast<size_t>(u) < graph.size()) {
                for (const auto& edge : graph[u]) {
                    int v = edge.first;
                    double weight = edge.second;
                    if (dists[u] + weight < dists[v] && dists[u] + weight < B) {
                        dists[v] = dists[u] + weight;
                        preds[v] = u;
                        K.push_back({dists[v], v});
                    }
                }
            }
        }
        D.batch_prepend(K);
    }

    VertexSet final_U = U;
    for (int x : W) {
        if (dists[x] < B_prime_final) {
            final_U.insert(x);
        }
    }

    return {B_prime_final, final_U};
}


// Main entry point for Python
std::map<int, double> bmss_p(const std::map<int, std::vector<std::pair<int, double>>>& adj_list, int source, int max_depth) {
    // --- NEW: Robustly determine the number of nodes ---
    int max_node_id = source;
    for(const auto& pair : adj_list) {
        max_node_id = std::max(max_node_id, pair.first);
        for(const auto& edge : pair.second) {
            max_node_id = std::max(max_node_id, edge.first);
        }
    }
    int n = max_node_id + 1;
    // ---------------------------------------------------

    Graph graph(n);
    for (const auto& pair : adj_list) {
        int u = pair.first;
        for (const auto& edge : pair.second) {
            graph[u].push_back(edge);
        }
    }

    int k = (n > 1) ? floor(pow(log2(n), 1.0/3.0)) : 1;
    int t = (n > 1) ? floor(pow(log2(n), 2.0/3.0)) : 1;
    k = std::max(1, k);
    t = std::max(1, t);

    Distances dists(n, std::numeric_limits<double>::infinity());
    Predecessors preds(n, -1);

    // Ensure source is within bounds before accessing
    if (source < n) {
        dists[source] = 0;
    } else {
         // Return an empty map if the source is invalid
        return {};
    }

    int l = (t > 0 && n > 1) ? ceil(log2(n) / t) : 1;
    double B = std::numeric_limits<double>::infinity();
    VertexSet S = {source};

    bmss_p_recursive(graph, l, B, S, k, t, dists, preds, max_depth, 0, l);

    std::map<int, double> result;
    for (int i = 0; i < n; ++i) {
        if (dists[i] != std::numeric_limits<double>::infinity()) {
            result[i] = dists[i];
        }
    }
    return result;
}


// Python Module Definition using pybind11
PYBIND11_MODULE(bmss_p_cpp, m) {
    m.doc() = "C++ implementation of BMSSP algorithm";
    m.def("bmss_p", &bmss_p, "A function that solves BMSSP",
          py::arg("adj_list"), py::arg("source"), py::arg("max_depth"));
}
