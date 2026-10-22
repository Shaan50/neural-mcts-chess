// Bitboard chess move generator used as the fast backend for search and data
// processing. Semantics intentionally mirror python-chess (legal-only en
// passant squares, cleaned castling rights, identical draw rules) so the two
// backends are interchangeable and can be cross-checked in tests.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace nmc {

using u64 = uint64_t;
using Move = uint16_t;  // from | to << 6 | promo << 12  (promo: 0 none, 1 N, 2 B, 3 R, 4 Q)

enum Color : int { WHITE = 0, BLACK = 1 };
enum PieceType : int { PAWN = 0, KNIGHT, BISHOP, ROOK, QUEEN, KING, NO_PIECE_TYPE };
enum Terminal : int { ONGOING = 0, CHECKMATE, STALEMATE, INSUFFICIENT, FIFTY_MOVES, THREEFOLD };

constexpr int POLICY_SIZE = 4672;
constexpr int NO_EP = 64;
extern const char* const START_FEN;

inline Move make_move(int from, int to, int promo = 0) { return Move(from | (to << 6) | (promo << 12)); }
inline int move_from(Move m) { return m & 63; }
inline int move_to(Move m) { return (m >> 6) & 63; }
inline int move_promo(Move m) { return (m >> 12) & 7; }

inline int lsb(u64 b) { return __builtin_ctzll(b); }
inline int popcount(u64 b) { return __builtin_popcountll(b); }
inline int pop_lsb(u64& b) {
  int s = lsb(b);
  b &= b - 1;
  return s;
}
inline Color operator!(Color c) { return Color(c ^ 1); }

void init_tables();  // idempotent; called by Position constructors

u64 knight_attacks(int sq);
u64 king_attacks(int sq);
u64 pawn_attacks(Color c, int sq);
u64 bishop_attacks(int sq, u64 occ);
u64 rook_attacks(int sq, u64 occ);

// AlphaZero 8x8x73 policy index of a move for the given side to move.
int policy_index(Move m, Color turn);

struct Undo {
  Move move;
  int8_t captured;  // piece type or -1
  int8_t ep;
  uint8_t castling;
  bool was_ep;
  int halfmove;
};

class Position {
 public:
  Position();
  explicit Position(const std::string& fen);

  void set_fen(const std::string& fen);
  std::string fen() const;

  Color turn() const { return stm_; }
  int halfmove_clock() const { return halfmove_; }
  int fullmove_number() const { return fullmove_; }
  int ep_square() const { return ep_; }
  uint8_t castling_rights() const { return castling_; }
  u64 hash() const { return hash_; }
  int ply() const { return int(stack_.size()); }
  int piece_type_at(int sq) const { return board_[sq] < 0 ? NO_PIECE_TYPE : board_[sq] % 6; }
  int color_at(int sq) const { return board_[sq] < 0 ? -1 : board_[sq] / 6; }
  u64 pieces(Color c, PieceType p) const { return bb_[c][p]; }

  void legal_moves(std::vector<Move>& out) const;
  std::vector<Move> legal_moves() const {
    std::vector<Move> v;
    legal_moves(v);
    return v;
  }
  bool has_legal_move() const;
  void push(Move m);
  void pop();

  bool in_check() const;
  bool insufficient_material() const;
  int repetition_count() const;  // occurrences of the current position, including now
  // True if the current position already occurred within the last `plies` plies.
  bool repeated_within(int plies) const;
  Terminal terminal() const;
  // Same as terminal() but reuses an already generated legal move list.
  Terminal terminal_given_moves(const std::vector<Move>& legal) const;

  void encode(int64_t* bb12, uint8_t* meta4) const;
  std::string move_to_uci(Move m) const;
  Move parse_uci(const std::string& s) const;  // throws std::invalid_argument when illegal
  Move parse_san(const std::string& s) const;  // throws std::invalid_argument when illegal/ambiguous
  const std::string& root_fen() const { return root_fen_; }
  std::vector<Move> move_stack() const;
  u64 perft(int depth);

 private:
  u64 bb_[2][6];
  u64 occ_[2];
  int8_t board_[64];  // -1 empty, else color * 6 + piece type
  Color stm_;
  uint8_t castling_;  // 1 = white O-O, 2 = white O-O-O, 4 = black O-O, 8 = black O-O-O
  int ep_;            // -1 unless an en-passant capture is legal
  int halfmove_;
  int fullmove_;
  u64 hash_;
  std::vector<Undo> stack_;
  std::vector<u64> hist_;  // hashes of every position so far, back() == hash_
  std::string root_fen_;

  void clear();
  void put(int sq, Color c, int pt);
  void remove(int sq);
  u64 compute_hash() const;
  u64 attackers_to(int sq, u64 occ, Color by) const;
  void pseudo_moves(std::vector<Move>& out) const;
  bool is_legal(Move m) const;  // for pseudo-legal non-castling moves
  bool ep_capture_legal() const;
  void clean_castling();
};

}  // namespace nmc
