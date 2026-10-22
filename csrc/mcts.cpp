#include "mcts.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace nmc {

void NNCache::insert(uint64_t key, Entry e) {
  if (capacity_ == 0) return;
  auto it = map_.find(key);
  if (it != map_.end()) {
    it->second = std::move(e);
    return;
  }
  while (map_.size() >= capacity_ && !order_.empty()) {
    map_.erase(order_.front());
    order_.pop_front();
  }
  map_.emplace(key, std::move(e));
  order_.push_back(key);
}

Search::Search(SearchParams params, std::shared_ptr<NNCache> cache)
    : params_(params), cache_(std::move(cache)), rng_(params.seed) {
  reset_tree();
}

void Search::reset_tree() {
  nodes_.clear();
  nodes_.emplace_back();
  root_ = 0;
  pending_.clear();
  noise_applied_ = false;
}

void Search::set_position(const Position& pos) {
  pos_ = pos;
  work_ = pos;
  reset_tree();
}

void Search::begin(bool add_noise) {
  want_noise_ = add_noise;
  noise_applied_ = false;
  max_depth_ = 0;
  depth_sum_ = depth_count_ = 0;
}

uint64_t Search::cache_key(const Position& p) const {
  // The network also sees the halfmove clock and repetition count.
  uint64_t extra = uint64_t(std::min(p.halfmove_clock(), 255)) | (uint64_t(std::min(p.repetition_count(), 3)) << 8);
  uint64_t z = extra * 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 31)) * 0xBF58476D1CE4E5B9ULL;
  return p.hash() ^ z;
}

void Search::advance(Move m) {
  if (!pending_.empty()) throw std::logic_error("advance() while evaluations are pending");
  pos_.push(m);
  work_ = pos_;
  const Node& r = nodes_[root_];
  int found = -1;
  if (r.state == EXPANDED)
    for (int i = 0; i < r.num_children; ++i)
      if (nodes_[r.first_child + i].move == m) found = r.first_child + i;
  if (found < 0 || nodes_[found].state == PENDING) {
    reset_tree();
    return;
  }
  // Copy the chosen subtree into a fresh arena (breadth first keeps siblings contiguous).
  std::vector<Node> fresh;
  fresh.reserve(nodes_.size() / 2 + 1);
  fresh.push_back(nodes_[found]);
  fresh[0].parent = -1;
  std::vector<std::pair<int, int>> queue{{found, 0}};
  for (size_t qi = 0; qi < queue.size(); ++qi) {
    const int old_idx = queue[qi].first, new_idx = queue[qi].second;
    const Node& o = nodes_[old_idx];
    if (o.num_children == 0) {
      fresh[new_idx].first_child = -1;
      continue;
    }
    const int first = int(fresh.size());
    fresh[new_idx].first_child = first;
    for (int i = 0; i < o.num_children; ++i) {
      Node c = nodes_[o.first_child + i];
      c.parent = new_idx;
      fresh.push_back(c);
      queue.emplace_back(o.first_child + i, first + i);
    }
  }
  nodes_.swap(fresh);
  root_ = 0;
  noise_applied_ = false;
  // A node scored as a draw by the in-tree two-fold rule is not terminal as a root.
  if (nodes_[0].state == TERMINAL && pos_.terminal() == ONGOING) reset_tree();
}

bool Search::done(int target_visits) const {
  const Node& r = nodes_[root_];
  return r.state == TERMINAL || r.n >= target_visits;
}

int Search::select_child(int node) const {
  const Node& p = nodes_[node];
  const int first = p.first_child, k = p.num_children;
  int child_visits = 0;
  double visited_policy = 0.0;
  for (int i = 0; i < k; ++i) {
    const Node& c = nodes_[first + i];
    const int cn = c.n + c.n_inflight;
    child_visits += cn;
    if (cn > 0) visited_policy += c.prior;
  }
  const double pn = double(child_visits);
  const double cpuct = params_.cpuct_init + std::log((pn + params_.cpuct_base + 1.0) / params_.cpuct_base);
  const double sqrt_n = std::sqrt(std::max(pn, 1.0));
  const double parent_q = p.n > 0 ? -p.w / p.n : 0.0;
  const double red = node == root_ ? params_.fpu_reduction_root : params_.fpu_reduction;
  const double fpu = parent_q - red * std::sqrt(visited_policy);
  double best = -std::numeric_limits<double>::infinity();
  int best_idx = first;
  for (int i = 0; i < k; ++i) {
    const Node& c = nodes_[first + i];
    const int cn = c.n + c.n_inflight;
    const double q = cn > 0 ? (c.w - params_.virtual_loss * c.n_inflight) / cn : fpu;
    const double u = cpuct * c.prior * sqrt_n / (1.0 + cn);
    if (q + u > best) {
      best = q + u;
      best_idx = first + i;
    }
  }
  return best_idx;
}

void Search::expand(int node, const std::vector<Move>& moves, const float* logits) {
  const int k = int(moves.size());
  const double t = params_.policy_temperature;
  double mx = -std::numeric_limits<double>::infinity();
  for (int i = 0; i < k; ++i) mx = std::max(mx, double(logits[i]));
  thread_local std::vector<double> p;
  p.assign(k, 0.0);
  double sum = 0.0;
  for (int i = 0; i < k; ++i) sum += (p[i] = std::exp((double(logits[i]) - mx) / t));
  const int first = int(nodes_.size());
  nodes_.resize(first + k);
  for (int i = 0; i < k; ++i) {
    Node& c = nodes_[first + i];
    c.parent = node;
    c.move = moves[i];
    c.prior = float(p[i] / sum);
  }
  Node& n = nodes_[node];
  n.first_child = first;
  n.num_children = uint16_t(k);
  n.state = EXPANDED;
}

void Search::backup(int node, double v, double d, bool from_inflight) {
  // v is the value for the side to move at `node`.
  for (int x = node; x != -1; x = nodes_[x].parent) {
    Node& nd = nodes_[x];
    nd.n += 1;
    nd.w += -v;
    nd.d += d;
    if (from_inflight) nd.n_inflight -= 1;
    v = -v;
  }
}

void Search::apply_noise() {
  Node& r = nodes_[root_];
  const int k = r.num_children;
  if (k == 0) return;
  std::gamma_distribution<double> gamma(params_.dirichlet_alpha, 1.0);
  std::vector<double> noise(k);
  double sum = 0.0;
  for (auto& x : noise) sum += (x = gamma(rng_));
  if (sum <= 0) return;
  const double eps = params_.dirichlet_epsilon;
  for (int i = 0; i < k; ++i) {
    Node& c = nodes_[r.first_child + i];
    c.prior = float((1.0 - eps) * c.prior + eps * noise[i] / sum);
  }
  noise_applied_ = true;
}

int Search::gather(Batch& batch, int max_leaves, int target_visits) {
  int collected = 0, collisions = 0;
  while (collected < max_leaves && collisions <= params_.max_collisions) {
    {
      const Node& r = nodes_[root_];
      if (r.state == TERMINAL || r.n + r.n_inflight >= target_visits) break;
    }
    int node = root_, depth = 0;
    while (nodes_[node].state == EXPANDED) {
      if (node == root_ && want_noise_ && !noise_applied_) apply_noise();
      const int c = select_child(node);
      work_.push(nodes_[c].move);
      ++depth;
      node = c;
    }
    max_depth_ = std::max(max_depth_, depth);
    depth_sum_ += depth;
    ++depth_count_;
    auto unwind = [&]() {
      for (int i = 0; i < depth; ++i) work_.pop();
    };

    const uint8_t state = nodes_[node].state;
    if (state == PENDING) {
      ++collisions;
      --depth_count_;
      depth_sum_ -= depth;
      unwind();
      continue;
    }
    if (state == TERMINAL) {
      const double v = nodes_[node].terminal_v;
      backup(node, v, v == 0.0 ? 1.0 : 0.0, false);
      unwind();
      continue;
    }

    // Unexpanded leaf: game over, cached, or queued for the network.
    work_.legal_moves(moves_buf_);
    Terminal t = work_.terminal_given_moves(moves_buf_);
    if (t == ONGOING && params_.twofold_draw && depth > 0 && work_.repeated_within(depth)) t = THREEFOLD;
    if (t != ONGOING) {
      Node& leaf = nodes_[node];
      leaf.state = TERMINAL;
      leaf.terminal_v = t == CHECKMATE ? -1.f : 0.f;
      backup(node, leaf.terminal_v, t == CHECKMATE ? 0.0 : 1.0, false);
      unwind();
      continue;
    }
    const uint64_t key = cache_key(work_);
    if (cache_) {
      ++cache_->lookups;
      const NNCache::Entry* e = cache_->find(key);
      if (e && e->logits.size() == moves_buf_.size()) {
        ++cache_->hits;
        const float wdl[3] = {e->wdl[0], e->wdl[1], e->wdl[2]};
        expand(node, moves_buf_, e->logits.data());
        backup(node, double(wdl[0]) - wdl[2], wdl[1], false);
        unwind();
        continue;
      }
    }
    nodes_[node].state = PENDING;
    for (int x = node; x != -1; x = nodes_[x].parent) nodes_[x].n_inflight += 1;
    const int row = batch.rows++;
    batch.bb.resize(size_t(row + 1) * 12);
    batch.meta.resize(size_t(row + 1) * 4);
    work_.encode(&batch.bb[size_t(row) * 12], &batch.meta[size_t(row) * 4]);
    const Color turn = work_.turn();
    for (Move m : moves_buf_) batch.legal.push_back(int64_t(row) * POLICY_SIZE + policy_index(m, turn));
    batch.n_legal.push_back(int32_t(moves_buf_.size()));
    pending_.push_back(Pending{node, key, moves_buf_});
    ++collected;
    unwind();
  }
  return collected;
}

void Search::apply(const float* legal_logits, const float* wdl, int* consumed_logits) {
  int off = 0;
  for (size_t i = 0; i < pending_.size(); ++i) {
    const Pending& p = pending_[i];
    const int k = int(p.moves.size());
    const float* lg = legal_logits + off;
    const float* w = wdl + 3 * i;
    if (cache_) cache_->insert(p.key, NNCache::Entry{std::vector<float>(lg, lg + k), {w[0], w[1], w[2]}});
    expand(p.node, p.moves, lg);
    backup(p.node, double(w[0]) - double(w[2]), w[1], true);
    off += k;
  }
  *consumed_logits = off;
  pending_.clear();
}

double Search::root_q() const {
  const Node& r = nodes_[root_];
  return r.n > 0 ? -r.w / r.n : 0.0;
}

double Search::root_d() const {
  const Node& r = nodes_[root_];
  return r.n > 0 ? r.d / r.n : 0.0;
}

std::vector<ChildInfo> Search::root_children() const {
  std::vector<ChildInfo> out;
  const Node& r = nodes_[root_];
  if (r.state != EXPANDED) return out;
  for (int i = 0; i < r.num_children; ++i) {
    const Node& c = nodes_[r.first_child + i];
    out.push_back(ChildInfo{c.move, policy_index(c.move, pos_.turn()), c.n, c.n > 0 ? c.w / c.n : 0.0, c.prior});
  }
  return out;
}

std::vector<Move> Search::pv(int max_len) const {
  std::vector<Move> out;
  int node = root_;
  while (int(out.size()) < max_len && nodes_[node].state == EXPANDED) {
    const Node& p = nodes_[node];
    int best = -1, best_n = 0;
    for (int i = 0; i < p.num_children; ++i)
      if (nodes_[p.first_child + i].n > best_n) best_n = nodes_[p.first_child + i].n, best = p.first_child + i;
    if (best < 0) break;
    out.push_back(nodes_[best].move);
    node = best;
  }
  return out;
}

}  // namespace nmc
