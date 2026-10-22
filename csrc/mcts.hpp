// Batched PUCT Monte-Carlo tree search.
//
// The tree lives in a flat arena (std::vector<Node>) with children stored
// contiguously. A Search object never calls the neural network itself:
// gather() walks the tree (applying virtual loss) until it reaches leaves that
// need an evaluation and writes their encodings into a batch; the caller runs
// the network on the batch and hands the results back through apply(). Many
// Search objects (one per concurrent game) share a batch, so a single GPU call
// serves hundreds of games.
#pragma once

#include <cstdint>
#include <deque>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

#include "chess.hpp"

namespace azc {

struct SearchParams {
  double cpuct_init = 1.25;     // AlphaZero: c(s) = c_init + log((N + c_base + 1) / c_base)
  double cpuct_base = 19652.0;
  double fpu_reduction = 0.25;  // unvisited child Q = parent Q - r * sqrt(visited policy mass)
  double fpu_reduction_root = 0.0;
  double policy_temperature = 1.0;
  double dirichlet_alpha = 0.3;
  double dirichlet_epsilon = 0.25;
  double virtual_loss = 1.0;
  int max_collisions = 8;       // per gather() call before giving up on more leaves
  bool twofold_draw = true;     // a repetition inside the search tree scores as a draw
  uint64_t seed = 0;
};

// FIFO-evicting cache of network outputs keyed by position (+ rule state).
class NNCache {
 public:
  explicit NNCache(size_t capacity) : capacity_(capacity) {}
  struct Entry {
    std::vector<float> logits;  // legal-move logits in generation order
    float wdl[3];
  };
  const Entry* find(uint64_t key) const {
    auto it = map_.find(key);
    return it == map_.end() ? nullptr : &it->second;
  }
  void insert(uint64_t key, Entry e);
  size_t size() const { return map_.size(); }
  uint64_t hits = 0, lookups = 0;
  void clear() {
    map_.clear();
    order_.clear();
  }

 private:
  size_t capacity_;
  std::unordered_map<uint64_t, Entry> map_;
  std::deque<uint64_t> order_;
};

enum NodeState : uint8_t { UNEXPANDED = 0, PENDING = 1, EXPANDED = 2, TERMINAL = 3 };

struct Node {
  int32_t parent = -1;
  int32_t first_child = -1;
  uint16_t num_children = 0;
  Move move = 0;
  float prior = 0.f;
  int32_t n = 0;           // completed visits
  int32_t n_inflight = 0;  // visits awaiting a network evaluation (virtual loss)
  double w = 0.0;          // value sum from the perspective of the player who played `move`
  double d = 0.0;          // draw-probability sum
  uint8_t state = UNEXPANDED;
  float terminal_v = 0.f;  // value for the side to move at this node when TERMINAL
};

struct ChildInfo {
  Move move;
  int policy_index;
  int n;
  double q;  // from the root side-to-move perspective
  float prior;
};

// Output buffers shared by all searches of one batch.
struct Batch {
  std::vector<int64_t> bb;        // rows * 12
  std::vector<uint8_t> meta;      // rows * 4
  std::vector<int64_t> legal;     // flat indices row * POLICY_SIZE + policy_index
  std::vector<int32_t> n_legal;   // per row
  int rows = 0;
  void clear() {
    bb.clear();
    meta.clear();
    legal.clear();
    n_legal.clear();
    rows = 0;
  }
};

class Search {
 public:
  Search(SearchParams params, std::shared_ptr<NNCache> cache);

  void set_position(const Position& pos);
  const Position& position() const { return pos_; }
  // Play a move at the root, keeping the corresponding subtree.
  void advance(Move m);
  // Called before each move's search; noise is added once the root is expanded.
  void begin(bool add_noise);

  // Walk the tree and append up to max_leaves leaves to `batch`, stopping once
  // root visits (completed + in flight) reach target_visits.
  int gather(Batch& batch, int max_leaves, int target_visits);
  // Consume this search's slice of network results (rows in gather order).
  // legal_logits points at the first logit of this search's first row.
  void apply(const float* legal_logits, const float* wdl, int* consumed_logits);

  int pending() const { return int(pending_.size()); }
  int pending_legal() const {
    int total = 0;
    for (const auto& p : pending_) total += int(p.moves.size());
    return total;
  }
  bool done(int target_visits) const;
  int root_visits() const { return nodes_[root_].n; }
  double root_q() const;  // side to move perspective
  double root_d() const;
  std::vector<ChildInfo> root_children() const;
  std::vector<Move> pv(int max_len) const;
  size_t tree_size() const { return nodes_.size(); }
  int max_depth() const { return max_depth_; }
  double mean_depth() const { return depth_count_ ? double(depth_sum_) / depth_count_ : 0.0; }
  SearchParams& params() { return params_; }

 private:
  struct Pending {
    int node;
    uint64_t key;
    std::vector<Move> moves;
  };

  SearchParams params_;
  std::shared_ptr<NNCache> cache_;
  Position pos_;   // root position
  Position work_;  // scratch position for descending the tree
  std::vector<Node> nodes_;
  int root_ = 0;
  std::vector<Pending> pending_;
  std::mt19937_64 rng_;
  bool want_noise_ = false, noise_applied_ = false;
  int max_depth_ = 0;
  int64_t depth_sum_ = 0, depth_count_ = 0;
  std::vector<int> path_;
  std::vector<Move> moves_buf_;

  void reset_tree();
  int select_child(int node) const;
  void expand(int node, const std::vector<Move>& moves, const float* logits);
  void backup(int node, double v, double d, bool from_inflight);
  void apply_noise();
  uint64_t cache_key(const Position& p) const;
};

}  // namespace azc
