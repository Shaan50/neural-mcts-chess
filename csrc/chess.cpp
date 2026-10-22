#include "chess.hpp"

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <sstream>
#include <stdexcept>

namespace azc {

const char* const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

namespace {

constexpr u64 RANK_1 = 0xFFULL;
constexpr u64 RANK_3 = RANK_1 << 16;
constexpr u64 RANK_6 = RANK_1 << 40;
constexpr u64 RANK_8 = RANK_1 << 56;
constexpr u64 DARK_SQUARES = 0xAA55AA55AA55AA55ULL;

u64 KNIGHT_ATT[64], KING_ATT[64], PAWN_ATT[2][64];
u64 RAYS[8][64];  // N, NE, E, SE, S, SW, W, NW
const int RAY_DF[8] = {0, 1, 1, 1, 0, -1, -1, -1};
const int RAY_DR[8] = {1, 1, 0, -1, -1, -1, 0, 1};
const bool RAY_POSITIVE[8] = {true, true, true, false, false, false, false, true};

int16_t MOVE_TABLE[64][64][4];  // [from][to][promo slot: none/queen, N, B, R] in oriented coords
uint8_t CASTLE_MASK[64];

u64 Z_PIECE[2][6][64], Z_CASTLE[16], Z_EP[8], Z_SIDE;

std::once_flag init_flag;

u64 splitmix64(u64& s) {
  u64 z = (s += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

bool on_board(int f, int r) { return f >= 0 && f < 8 && r >= 0 && r < 8; }

int sign(int x) { return (x > 0) - (x < 0); }

// Mirrors azchess/encoding/moves.py::_plane.
int move_plane(int df, int dr, int promo_slot) {
  static const int KN[8][2] = {{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};
  if (promo_slot >= 1 && promo_slot <= 3) {
    if (dr != 1 || df < -1 || df > 1) return -1;
    return 64 + (df + 1) * 3 + (promo_slot - 1);
  }
  for (int i = 0; i < 8; ++i)
    if (KN[i][0] == df && KN[i][1] == dr) return 56 + i;
  if (df == 0 && dr == 0) return -1;
  if (df != 0 && dr != 0 && std::abs(df) != std::abs(dr)) return -1;
  int dist = std::max(std::abs(df), std::abs(dr));
  for (int d = 0; d < 8; ++d)
    if (RAY_DF[d] == sign(df) && RAY_DR[d] == sign(dr)) return d * 7 + dist - 1;
  return -1;
}

void do_init() {
  for (int sq = 0; sq < 64; ++sq) {
    int f = sq & 7, r = sq >> 3;
    static const int KN[8][2] = {{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};
    KNIGHT_ATT[sq] = KING_ATT[sq] = 0;
    for (auto& k : KN)
      if (on_board(f + k[0], r + k[1])) KNIGHT_ATT[sq] |= 1ULL << ((r + k[1]) * 8 + f + k[0]);
    for (int df = -1; df <= 1; ++df)
      for (int dr = -1; dr <= 1; ++dr)
        if ((df || dr) && on_board(f + df, r + dr)) KING_ATT[sq] |= 1ULL << ((r + dr) * 8 + f + df);
    PAWN_ATT[WHITE][sq] = PAWN_ATT[BLACK][sq] = 0;
    for (int df : {-1, 1}) {
      if (on_board(f + df, r + 1)) PAWN_ATT[WHITE][sq] |= 1ULL << ((r + 1) * 8 + f + df);
      if (on_board(f + df, r - 1)) PAWN_ATT[BLACK][sq] |= 1ULL << ((r - 1) * 8 + f + df);
    }
    for (int d = 0; d < 8; ++d) {
      u64 ray = 0;
      int ff = f + RAY_DF[d], rr = r + RAY_DR[d];
      while (on_board(ff, rr)) {
        ray |= 1ULL << (rr * 8 + ff);
        ff += RAY_DF[d];
        rr += RAY_DR[d];
      }
      RAYS[d][sq] = ray;
    }
  }
  for (int from = 0; from < 64; ++from)
    for (int to = 0; to < 64; ++to)
      for (int slot = 0; slot < 4; ++slot) {
        const bool promo_ok = slot == 0 || ((from >> 3) == 6 && (to >> 3) == 7);
        int plane = promo_ok ? move_plane((to & 7) - (from & 7), (to >> 3) - (from >> 3), slot) : -1;
        MOVE_TABLE[from][to][slot] = int16_t(plane < 0 ? -1 : plane * 64 + from);
      }
  for (int sq = 0; sq < 64; ++sq) CASTLE_MASK[sq] = 15;
  CASTLE_MASK[4] &= ~3;    // e1
  CASTLE_MASK[7] &= ~1;    // h1
  CASTLE_MASK[0] &= ~2;    // a1
  CASTLE_MASK[60] &= ~12;  // e8
  CASTLE_MASK[63] &= ~4;   // h8
  CASTLE_MASK[56] &= ~8;   // a8

  u64 seed = 0x5EED1234ABCDULL;
  for (auto& c : Z_PIECE)
    for (auto& p : c)
      for (auto& s : p) s = splitmix64(seed);
  for (auto& c : Z_CASTLE) c = splitmix64(seed);
  for (auto& e : Z_EP) e = splitmix64(seed);
  Z_SIDE = splitmix64(seed);
}

inline u64 slide(int sq, u64 occ, int d) {
  u64 att = RAYS[d][sq];
  u64 blockers = att & occ;
  if (blockers) {
    int b = RAY_POSITIVE[d] ? lsb(blockers) : 63 - __builtin_clzll(blockers);
    att ^= RAYS[d][b];
  }
  return att;
}

inline u64 sq_bb(int sq) { return 1ULL << sq; }

}  // namespace

void init_tables() { std::call_once(init_flag, do_init); }

u64 knight_attacks(int sq) { return KNIGHT_ATT[sq]; }
u64 king_attacks(int sq) { return KING_ATT[sq]; }
u64 pawn_attacks(Color c, int sq) { return PAWN_ATT[c][sq]; }
u64 bishop_attacks(int sq, u64 occ) { return slide(sq, occ, 1) | slide(sq, occ, 3) | slide(sq, occ, 5) | slide(sq, occ, 7); }
u64 rook_attacks(int sq, u64 occ) { return slide(sq, occ, 0) | slide(sq, occ, 2) | slide(sq, occ, 4) | slide(sq, occ, 6); }

int policy_index(Move m, Color turn) {
  int from = move_from(m), to = move_to(m), promo = move_promo(m);
  if (turn == BLACK) {
    from ^= 56;
    to ^= 56;
  }
  int slot = (promo == KNIGHT || promo == BISHOP || promo == ROOK) ? promo : 0;
  return MOVE_TABLE[from][to][slot];
}

// ---------------------------------------------------------------------------
// Position

Position::Position() { set_fen(START_FEN); }
Position::Position(const std::string& fen) { set_fen(fen); }

void Position::clear() {
  for (auto& c : bb_)
    for (auto& p : c) p = 0;
  occ_[0] = occ_[1] = 0;
  for (auto& b : board_) b = -1;
  stm_ = WHITE;
  castling_ = 0;
  ep_ = -1;
  halfmove_ = 0;
  fullmove_ = 1;
  stack_.clear();
  hist_.clear();
}

void Position::put(int sq, Color c, int pt) {
  bb_[c][pt] |= sq_bb(sq);
  occ_[c] |= sq_bb(sq);
  board_[sq] = int8_t(c * 6 + pt);
}

void Position::remove(int sq) {
  int code = board_[sq];
  if (code < 0) return;
  Color c = Color(code / 6);
  bb_[c][code % 6] &= ~sq_bb(sq);
  occ_[c] &= ~sq_bb(sq);
  board_[sq] = -1;
}

void Position::set_fen(const std::string& fen) {
  init_tables();
  clear();
  std::istringstream ss(fen);
  std::string placement, side, castle = "-", ep = "-";
  int hm = 0, fm = 1;
  if (!(ss >> placement >> side)) throw std::invalid_argument("bad FEN: " + fen);
  ss >> castle >> ep;
  if (!(ss >> hm)) hm = 0;
  if (!(ss >> fm)) fm = 1;

  int rank = 7, file = 0;
  for (char ch : placement) {
    if (ch == '/') {
      --rank;
      file = 0;
    } else if (ch >= '1' && ch <= '8') {
      file += ch - '0';
    } else {
      const std::string pcs = "pnbrqk";
      auto pos = pcs.find(char(std::tolower(ch)));
      if (pos == std::string::npos || rank < 0 || file > 7) throw std::invalid_argument("bad FEN: " + fen);
      put(rank * 8 + file, std::isupper(ch) ? WHITE : BLACK, int(pos));
      ++file;
    }
  }
  if (popcount(bb_[WHITE][KING]) != 1 || popcount(bb_[BLACK][KING]) != 1) throw std::invalid_argument("FEN needs one king per side: " + fen);
  stm_ = side == "b" ? BLACK : WHITE;
  for (char ch : castle) {
    if (ch == 'K') castling_ |= 1;
    if (ch == 'Q') castling_ |= 2;
    if (ch == 'k') castling_ |= 4;
    if (ch == 'q') castling_ |= 8;
  }
  clean_castling();
  halfmove_ = hm;
  fullmove_ = std::max(fm, 1);
  ep_ = -1;
  if (ep != "-" && ep.size() == 2) {
    int sq = (ep[1] - '1') * 8 + (ep[0] - 'a');
    int pawn_sq = stm_ == WHITE ? sq - 8 : sq + 8;
    bool shape_ok = sq >= 0 && sq < 64 && board_[sq] < 0 && pawn_sq >= 0 && pawn_sq < 64 &&
                    board_[pawn_sq] == int8_t((!stm_) * 6 + PAWN) && ((stm_ == WHITE) ? (sq >> 3) == 5 : (sq >> 3) == 2);
    if (shape_ok) {
      ep_ = sq;
      if (!ep_capture_legal()) ep_ = -1;
    }
  }
  hash_ = compute_hash();
  hist_.push_back(hash_);
  root_fen_ = this->fen();
}

void Position::clean_castling() {
  auto has = [&](int sq, Color c, int pt) { return board_[sq] == int8_t(c * 6 + pt); };
  if (!has(4, WHITE, KING)) castling_ &= ~3;
  if (!has(7, WHITE, ROOK)) castling_ &= ~1;
  if (!has(0, WHITE, ROOK)) castling_ &= ~2;
  if (!has(60, BLACK, KING)) castling_ &= ~12;
  if (!has(63, BLACK, ROOK)) castling_ &= ~4;
  if (!has(56, BLACK, ROOK)) castling_ &= ~8;
}

std::string Position::fen() const {
  std::string out;
  const char* syms = "PNBRQKpnbrqk";
  for (int r = 7; r >= 0; --r) {
    int empty = 0;
    for (int f = 0; f < 8; ++f) {
      int code = board_[r * 8 + f];
      if (code < 0) {
        ++empty;
        continue;
      }
      if (empty) out += char('0' + empty), empty = 0;
      out += syms[code];
    }
    if (empty) out += char('0' + empty);
    if (r) out += '/';
  }
  out += stm_ == WHITE ? " w " : " b ";
  std::string c;
  if (castling_ & 1) c += 'K';
  if (castling_ & 2) c += 'Q';
  if (castling_ & 4) c += 'k';
  if (castling_ & 8) c += 'q';
  out += c.empty() ? "-" : c;
  out += ' ';
  if (ep_ >= 0) {
    out += char('a' + (ep_ & 7));
    out += char('1' + (ep_ >> 3));
  } else {
    out += '-';
  }
  out += ' ' + std::to_string(halfmove_) + ' ' + std::to_string(fullmove_);
  return out;
}

u64 Position::compute_hash() const {
  u64 h = 0;
  for (int c = 0; c < 2; ++c)
    for (int p = 0; p < 6; ++p) {
      u64 b = bb_[c][p];
      while (b) h ^= Z_PIECE[c][p][pop_lsb(b)];
    }
  h ^= Z_CASTLE[castling_];
  if (ep_ >= 0) h ^= Z_EP[ep_ & 7];
  if (stm_ == BLACK) h ^= Z_SIDE;
  return h;
}

u64 Position::attackers_to(int sq, u64 occ, Color by) const {
  return (pawn_attacks(!by, sq) & bb_[by][PAWN]) | (knight_attacks(sq) & bb_[by][KNIGHT]) |
         (king_attacks(sq) & bb_[by][KING]) | (bishop_attacks(sq, occ) & (bb_[by][BISHOP] | bb_[by][QUEEN])) |
         (rook_attacks(sq, occ) & (bb_[by][ROOK] | bb_[by][QUEEN]));
}

bool Position::in_check() const {
  return attackers_to(lsb(bb_[stm_][KING]), occ_[0] | occ_[1], !stm_) != 0;
}

void Position::pseudo_moves(std::vector<Move>& out) const {
  const Color us = stm_, them = !stm_;
  const u64 own = occ_[us], enemy = occ_[them], occ = own | enemy, empty = ~occ;

  // Pawns
  const u64 pawns = bb_[us][PAWN];
  const int up = us == WHITE ? 8 : -8;
  const u64 promo_rank = us == WHITE ? RANK_8 : RANK_1;
  auto add_pawn = [&](int from, int to) {
    if (sq_bb(to) & promo_rank) {
      for (int p : {QUEEN, ROOK, BISHOP, KNIGHT}) out.push_back(make_move(from, to, p));
    } else {
      out.push_back(make_move(from, to));
    }
  };
  u64 single = (us == WHITE ? pawns << 8 : pawns >> 8) & empty;
  u64 dbl = (us == WHITE ? (single & RANK_3) << 8 : (single & RANK_6) >> 8) & empty;
  for (u64 b = single; b;) {
    int to = pop_lsb(b);
    add_pawn(to - up, to);
  }
  for (u64 b = dbl; b;) {
    int to = pop_lsb(b);
    out.push_back(make_move(to - 2 * up, to));
  }
  for (u64 b = pawns; b;) {
    int from = pop_lsb(b);
    u64 att = pawn_attacks(us, from);
    for (u64 t = att & enemy; t;) add_pawn(from, pop_lsb(t));
    if (ep_ >= 0 && (att & sq_bb(ep_))) out.push_back(make_move(from, ep_));
  }

  // Pieces
  for (u64 b = bb_[us][KNIGHT]; b;) {
    int from = pop_lsb(b);
    for (u64 t = knight_attacks(from) & ~own; t;) out.push_back(make_move(from, pop_lsb(t)));
  }
  for (u64 b = bb_[us][BISHOP] | bb_[us][QUEEN]; b;) {
    int from = pop_lsb(b);
    for (u64 t = bishop_attacks(from, occ) & ~own; t;) out.push_back(make_move(from, pop_lsb(t)));
  }
  for (u64 b = bb_[us][ROOK] | bb_[us][QUEEN]; b;) {
    int from = pop_lsb(b);
    for (u64 t = rook_attacks(from, occ) & ~own; t;) out.push_back(make_move(from, pop_lsb(t)));
  }
  const int ksq = lsb(bb_[us][KING]);
  for (u64 t = king_attacks(ksq) & ~own; t;) out.push_back(make_move(ksq, pop_lsb(t)));

  // Castling (fully legality-checked here)
  const uint8_t k_right = us == WHITE ? 1 : 4, q_right = us == WHITE ? 2 : 8;
  if ((castling_ & (k_right | q_right)) && !attackers_to(ksq, occ, them)) {
    const int base = us == WHITE ? 0 : 56;
    if ((castling_ & k_right) && !(occ & (sq_bb(base + 5) | sq_bb(base + 6))) && !attackers_to(base + 5, occ, them) &&
        !attackers_to(base + 6, occ, them))
      out.push_back(make_move(base + 4, base + 6));
    if ((castling_ & q_right) && !(occ & (sq_bb(base + 1) | sq_bb(base + 2) | sq_bb(base + 3))) &&
        !attackers_to(base + 3, occ, them) && !attackers_to(base + 2, occ, them))
      out.push_back(make_move(base + 4, base + 2));
  }
}

bool Position::is_legal(Move m) const {
  const Color us = stm_, them = !stm_;
  const int from = move_from(m), to = move_to(m);
  const int pt = board_[from] % 6;
  if (pt == KING && std::abs(to - from) == 2) return true;  // castling already checked

  u64 occ = ((occ_[0] | occ_[1]) & ~sq_bb(from)) | sq_bb(to);
  u64 e[6];
  for (int p = 0; p < 6; ++p) e[p] = bb_[them][p] & ~sq_bb(to);
  if (pt == PAWN && to == ep_) {
    int cap = us == WHITE ? to - 8 : to + 8;
    e[PAWN] &= ~sq_bb(cap);
    occ &= ~sq_bb(cap);
  }
  const int ksq = pt == KING ? to : lsb(bb_[us][KING]);
  if (pawn_attacks(us, ksq) & e[PAWN]) return false;
  if (knight_attacks(ksq) & e[KNIGHT]) return false;
  if (king_attacks(ksq) & e[KING]) return false;
  if (bishop_attacks(ksq, occ) & (e[BISHOP] | e[QUEEN])) return false;
  if (rook_attacks(ksq, occ) & (e[ROOK] | e[QUEEN])) return false;
  return true;
}

void Position::legal_moves(std::vector<Move>& out) const {
  out.clear();
  thread_local std::vector<Move> pseudo;
  pseudo.clear();
  pseudo_moves(pseudo);
  for (Move m : pseudo)
    if (is_legal(m)) out.push_back(m);
}

bool Position::has_legal_move() const {
  thread_local std::vector<Move> pseudo;
  pseudo.clear();
  pseudo_moves(pseudo);
  for (Move m : pseudo)
    if (is_legal(m)) return true;
  return false;
}

bool Position::ep_capture_legal() const {
  // stm_ is the side that may capture onto ep_.
  for (u64 b = pawn_attacks(!stm_, ep_) & bb_[stm_][PAWN]; b;)
    if (is_legal(make_move(pop_lsb(b), ep_))) return true;
  return false;
}

void Position::push(Move m) {
  const Color us = stm_, them = !stm_;
  const int from = move_from(m), to = move_to(m), promo = move_promo(m);
  const int pt = board_[from] % 6;
  Undo u{m, -1, int8_t(ep_), castling_, false, halfmove_};

  int cap_sq = to;
  if (pt == PAWN && to == ep_) {
    cap_sq = us == WHITE ? to - 8 : to + 8;
    u.was_ep = true;
  }
  if (board_[cap_sq] >= 0) {
    u.captured = int8_t(board_[cap_sq] % 6);
    remove(cap_sq);
  }
  remove(from);
  put(to, us, promo ? promo : pt);
  if (pt == KING && std::abs(to - from) == 2) {
    if (to > from) {
      remove(from + 3);
      put(from + 1, us, ROOK);
    } else {
      remove(from - 4);
      put(from - 1, us, ROOK);
    }
  }
  castling_ &= CASTLE_MASK[from] & CASTLE_MASK[to];
  halfmove_ = (pt == PAWN || u.captured >= 0) ? 0 : halfmove_ + 1;
  if (us == BLACK) ++fullmove_;
  stm_ = them;
  ep_ = -1;
  if (pt == PAWN && std::abs(to - from) == 16) {
    ep_ = (from + to) / 2;
    if (!ep_capture_legal()) ep_ = -1;
  }
  hash_ = compute_hash();
  stack_.push_back(u);
  hist_.push_back(hash_);
}

void Position::pop() {
  if (stack_.empty()) throw std::out_of_range("pop() on a position without moves");
  const Undo u = stack_.back();
  stack_.pop_back();
  hist_.pop_back();
  stm_ = !stm_;
  const Color us = stm_, them = !stm_;
  const int from = move_from(u.move), to = move_to(u.move), promo = move_promo(u.move);
  const int pt = promo ? PAWN : board_[to] % 6;
  remove(to);
  put(from, us, pt);
  if (pt == KING && std::abs(to - from) == 2) {
    if (to > from) {
      remove(from + 1);
      put(from + 3, us, ROOK);
    } else {
      remove(from - 1);
      put(from - 4, us, ROOK);
    }
  }
  if (u.captured >= 0) put(u.was_ep ? (us == WHITE ? to - 8 : to + 8) : to, them, u.captured);
  ep_ = u.ep;
  castling_ = u.castling;
  halfmove_ = u.halfmove;
  if (us == BLACK) --fullmove_;
  hash_ = hist_.back();
}

bool Position::insufficient_material() const {
  // Port of python-chess Board.has_insufficient_material for both colours.
  const u64 all_pawns = bb_[0][PAWN] | bb_[1][PAWN];
  const u64 all_knights = bb_[0][KNIGHT] | bb_[1][KNIGHT];
  const u64 all_bishops = bb_[0][BISHOP] | bb_[1][BISHOP];
  for (int ci = 0; ci < 2; ++ci) {
    const Color c = Color(ci);
    const u64 own = occ_[c];
    if (own & (bb_[c][PAWN] | bb_[c][ROOK] | bb_[c][QUEEN])) return false;
    if (own & bb_[c][KNIGHT]) {
      if (!(popcount(own) <= 2 && !(occ_[!c] & ~bb_[!c][KING] & ~bb_[!c][QUEEN]))) return false;
      continue;
    }
    if (own & bb_[c][BISHOP]) {
      const bool same_color = !(all_bishops & DARK_SQUARES) || !(all_bishops & ~DARK_SQUARES);
      if (!(same_color && !all_pawns && !all_knights)) return false;
    }
  }
  return true;
}

int Position::repetition_count() const {
  int count = 0;
  const int n = int(hist_.size());
  const int lo = std::max(0, n - 1 - halfmove_);
  for (int i = n - 1; i >= lo; i -= 2)
    if (hist_[i] == hash_) ++count;
  return count;
}

bool Position::repeated_within(int plies) const {
  const int n = int(hist_.size());
  const int lo = std::max({0, n - 1 - plies, n - 1 - halfmove_});
  for (int i = n - 3; i >= lo; i -= 2)
    if (hist_[i] == hash_) return true;
  return false;
}

std::vector<Move> Position::move_stack() const {
  std::vector<Move> out;
  out.reserve(stack_.size());
  for (const auto& u : stack_) out.push_back(u.move);
  return out;
}

Terminal Position::terminal_given_moves(const std::vector<Move>& legal) const {
  if (legal.empty()) return in_check() ? CHECKMATE : STALEMATE;
  if (insufficient_material()) return INSUFFICIENT;
  if (halfmove_ >= 100) return FIFTY_MOVES;
  if (repetition_count() >= 3) return THREEFOLD;
  return ONGOING;
}

Terminal Position::terminal() const {
  if (!has_legal_move()) return in_check() ? CHECKMATE : STALEMATE;
  if (insufficient_material()) return INSUFFICIENT;
  if (halfmove_ >= 100) return FIFTY_MOVES;
  if (repetition_count() >= 3) return THREEFOLD;
  return ONGOING;
}

void Position::encode(int64_t* bb12, uint8_t* meta4) const {
  const Color us = stm_, them = !stm_;
  for (int p = 0; p < 6; ++p) {
    u64 ours = bb_[us][p], theirs = bb_[them][p];
    if (us == BLACK) {
      ours = __builtin_bswap64(ours);
      theirs = __builtin_bswap64(theirs);
    }
    bb12[p] = int64_t(ours);
    bb12[6 + p] = int64_t(theirs);
  }
  meta4[0] = uint8_t(ep_ < 0 ? NO_EP : (us == WHITE ? ep_ : ep_ ^ 56));
  uint8_t ok, oq, tk, tq;
  if (us == WHITE) {
    ok = castling_ & 1, oq = (castling_ >> 1) & 1, tk = (castling_ >> 2) & 1, tq = (castling_ >> 3) & 1;
  } else {
    ok = (castling_ >> 2) & 1, oq = (castling_ >> 3) & 1, tk = castling_ & 1, tq = (castling_ >> 1) & 1;
  }
  meta4[1] = uint8_t(ok | (oq << 1) | (tk << 2) | (tq << 3) | ((us == BLACK) << 4));
  meta4[2] = uint8_t(std::min(halfmove_, 255));
  meta4[3] = uint8_t(std::min(repetition_count() - 1, 2));
}

std::string Position::move_to_uci(Move m) const {
  std::string s;
  int from = move_from(m), to = move_to(m), promo = move_promo(m);
  s += char('a' + (from & 7));
  s += char('1' + (from >> 3));
  s += char('a' + (to & 7));
  s += char('1' + (to >> 3));
  if (promo) s += "pnbrqk"[promo];
  return s;
}

Move Position::parse_uci(const std::string& s) const {
  if (s.size() < 4) throw std::invalid_argument("bad uci move: " + s);
  int from = (s[1] - '1') * 8 + (s[0] - 'a');
  int to = (s[3] - '1') * 8 + (s[2] - 'a');
  int promo = 0;
  if (s.size() >= 5) {
    const std::string pcs = "pnbrqk";
    auto p = pcs.find(char(std::tolower(s[4])));
    if (p == std::string::npos) throw std::invalid_argument("bad uci move: " + s);
    promo = int(p);
  }
  for (Move m : legal_moves())
    if (move_from(m) == from && move_to(m) == to && move_promo(m) == promo) return m;
  throw std::invalid_argument("illegal move " + s + " in " + fen());
}

Move Position::parse_san(const std::string& san_in) const {
  std::string s = san_in;
  while (!s.empty() && (s.back() == '+' || s.back() == '#' || s.back() == '!' || s.back() == '?')) s.pop_back();
  std::vector<Move> legal = legal_moves();
  const int base = stm_ == WHITE ? 0 : 56;
  if (s == "O-O" || s == "0-0" || s == "O-O-O" || s == "0-0-0") {
    const int to = base + (s.size() == 3 ? 6 : 2);
    for (Move m : legal)
      if (move_from(m) == base + 4 && move_to(m) == to && board_[base + 4] == int8_t(stm_ * 6 + KING)) return m;
    throw std::invalid_argument("illegal castling " + san_in + " in " + fen());
  }
  int promo = 0;
  auto eq = s.find('=');
  if (eq != std::string::npos) {
    if (eq + 1 >= s.size()) throw std::invalid_argument("bad SAN " + san_in);
    promo = int(std::string("PNBRQK").find(s[eq + 1]));
    s = s.substr(0, eq);
  } else if (s.size() >= 3 && std::string("NBRQ").find(s.back()) != std::string::npos && std::islower(s[0])) {
    promo = int(std::string("PNBRQK").find(s.back()));  // "e8Q" style
    s.pop_back();
  }
  int pt = PAWN;
  size_t i = 0;
  if (!s.empty() && std::isupper(s[0])) {
    auto p = std::string("PNBRQK").find(s[0]);
    if (p == std::string::npos) throw std::invalid_argument("bad SAN " + san_in);
    pt = int(p);
    i = 1;
  }
  if (s.size() < i + 2) throw std::invalid_argument("bad SAN " + san_in);
  const int to = (s[s.size() - 1] - '1') * 8 + (s[s.size() - 2] - 'a');
  int from_file = -1, from_rank = -1;
  for (size_t j = i; j + 2 < s.size(); ++j) {
    char c = s[j];
    if (c >= 'a' && c <= 'h') from_file = c - 'a';
    else if (c >= '1' && c <= '8') from_rank = c - '1';
  }
  Move found = 0;
  int matches = 0;
  for (Move m : legal) {
    const int f = move_from(m);
    if (move_to(m) != to || board_[f] % 6 != pt || move_promo(m) != promo) continue;
    if (from_file >= 0 && (f & 7) != from_file) continue;
    if (from_rank >= 0 && (f >> 3) != from_rank) continue;
    found = m;
    ++matches;
  }
  if (matches != 1) throw std::invalid_argument((matches ? "ambiguous SAN " : "illegal SAN ") + san_in + " in " + fen());
  return found;
}

u64 Position::perft(int depth) {
  if (depth == 0) return 1;
  std::vector<Move> moves;
  legal_moves(moves);
  if (depth == 1) return moves.size();
  u64 total = 0;
  for (Move m : moves) {
    push(m);
    total += perft(depth - 1);
    pop();
  }
  return total;
}

}  // namespace azc
