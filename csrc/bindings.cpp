// pybind11 bindings: azchess._core
#include <cstring>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "chess.hpp"
#include "mcts.hpp"

namespace py = pybind11;
using namespace azc;

namespace {

template <typename T>
py::array_t<T> to_array(const std::vector<T>& v, std::vector<py::ssize_t> shape) {
  py::array_t<T> out(shape);
  if (!v.empty()) std::memcpy(out.mutable_data(), v.data(), v.size() * sizeof(T));
  return out;
}

py::tuple encode_position(const Position& p) {
  py::array_t<uint64_t> bb(12);
  py::array_t<uint8_t> meta(4);
  p.encode(reinterpret_cast<int64_t*>(bb.mutable_data()), meta.mutable_data());
  return py::make_tuple(bb, meta);
}

// Replay a game given as SAN tokens and return training arrays for every
// position before a move: (bb[N,12] uint64, meta[N,4] uint8, move_index[N] int16,
// legal_index[sum(n_legal)] int16, n_legal[N] int32, complete flag, termination code).
py::tuple encode_game(const std::string& fen, const std::vector<std::string>& sans) {
  Position pos(fen);
  const size_t n = sans.size();
  std::vector<int64_t> bb(n * 12);
  std::vector<uint8_t> meta(n * 4);
  std::vector<int32_t> n_legal(n);
  std::vector<int16_t> move_idx(n), legal;
  legal.reserve(n * 40);
  std::vector<Move> moves;
  size_t done = 0;
  {
    py::gil_scoped_release release;
    for (; done < n; ++done) {
      Move m;
      try {
        m = pos.parse_san(sans[done]);
      } catch (const std::invalid_argument&) {
        break;
      }
      pos.encode(&bb[done * 12], &meta[done * 4]);
      move_idx[done] = int16_t(policy_index(m, pos.turn()));
      pos.legal_moves(moves);
      n_legal[done] = int32_t(moves.size());
      for (Move lm : moves) legal.push_back(int16_t(policy_index(lm, pos.turn())));
      pos.push(m);
    }
  }
  bb.resize(done * 12);
  meta.resize(done * 4);
  move_idx.resize(done);
  n_legal.resize(done);
  auto bb_arr = to_array(bb, {py::ssize_t(done), 12});
  return py::make_tuple(bb_arr.attr("view")("uint64"), to_array(meta, {py::ssize_t(done), 4}), to_array(move_idx, {py::ssize_t(done)}),
                        to_array(legal, {py::ssize_t(legal.size())}), to_array(n_legal, {py::ssize_t(done)}), int(done == n),
                        int(pos.terminal()));
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "C++ chess move generation and batched MCTS for azchess";
  init_tables();
  m.attr("POLICY_SIZE") = POLICY_SIZE;
  m.attr("STARTING_FEN") = START_FEN;

  py::class_<Position>(m, "Position")
      .def(py::init<>())
      .def(py::init<const std::string&>(), py::arg("fen"))
      .def("copy", [](const Position& p) { return Position(p); })
      .def("__copy__", [](const Position& p) { return Position(p); })
      .def("__deepcopy__", [](const Position& p, py::dict) { return Position(p); })
      .def_property_readonly_static("backend", [](py::object) { return "cpp"; })
      .def_property_readonly("turn", [](const Position& p) { return p.turn() == WHITE; })
      .def("legal_moves", [](const Position& p) { return p.legal_moves(); })
      .def("policy_indices",
           [](const Position& p, const std::vector<Move>& moves) {
             py::array_t<int64_t> out(moves.size());
             auto r = out.mutable_unchecked<1>();
             for (size_t i = 0; i < moves.size(); ++i) r(i) = policy_index(moves[i], p.turn());
             return out;
           })
      .def("push", &Position::push, py::arg("move"), "Play a move (must be legal; unchecked).")
      .def("pop", [](Position& p) {
        auto st = p.move_stack();
        if (st.empty()) throw py::index_error("pop from empty move stack");
        p.pop();
        return st.back();
      })
      .def("push_uci", [](Position& p, const std::string& s) { p.push(p.parse_uci(s)); })
      .def("push_san", [](Position& p, const std::string& s) { p.push(p.parse_san(s)); })
      .def("parse_uci", &Position::parse_uci)
      .def("parse_san", &Position::parse_san)
      .def("uci", &Position::move_to_uci)
      .def("terminal", [](const Position& p) { return int(p.terminal()); })
      .def("is_check", &Position::in_check)
      .def("is_insufficient_material", &Position::insufficient_material)
      .def("repetition_count", &Position::repetition_count)
      .def("repeated_within", &Position::repeated_within)
      .def("ply", &Position::ply)
      .def("fen", &Position::fen)
      .def("root_fen", &Position::root_fen)
      .def("move_stack", &Position::move_stack)
      .def("hash", &Position::hash)
      .def("halfmove_clock", &Position::halfmove_clock)
      .def("fullmove_number", &Position::fullmove_number)
      .def("encode", &encode_position)
      .def("perft", &Position::perft)
      .def("piece_type_at", &Position::piece_type_at)
      .def("color_at", &Position::color_at)
      .def("__repr__", [](const Position& p) { return "<azchess._core.Position '" + p.fen() + "'>"; })
      .def(py::pickle([](const Position& p) { return py::make_tuple(p.root_fen(), p.move_stack()); },
                      [](py::tuple t) {
                        Position p(t[0].cast<std::string>());
                        for (Move mv : t[1].cast<std::vector<Move>>()) p.push(mv);
                        return p;
                      }));

  m.def("policy_index", [](Move mv, bool white) { return policy_index(mv, white ? WHITE : BLACK); });
  m.def("encode_game", &encode_game, py::arg("fen"), py::arg("sans"));

  py::class_<SearchParams>(m, "SearchParams")
      .def(py::init<>())
      .def_readwrite("cpuct_init", &SearchParams::cpuct_init)
      .def_readwrite("cpuct_base", &SearchParams::cpuct_base)
      .def_readwrite("fpu_reduction", &SearchParams::fpu_reduction)
      .def_readwrite("fpu_reduction_root", &SearchParams::fpu_reduction_root)
      .def_readwrite("policy_temperature", &SearchParams::policy_temperature)
      .def_readwrite("dirichlet_alpha", &SearchParams::dirichlet_alpha)
      .def_readwrite("dirichlet_epsilon", &SearchParams::dirichlet_epsilon)
      .def_readwrite("virtual_loss", &SearchParams::virtual_loss)
      .def_readwrite("max_collisions", &SearchParams::max_collisions)
      .def_readwrite("twofold_draw", &SearchParams::twofold_draw)
      .def_readwrite("seed", &SearchParams::seed);

  py::class_<NNCache, std::shared_ptr<NNCache>>(m, "NNCache")
      .def(py::init<size_t>(), py::arg("capacity"))
      .def("__len__", &NNCache::size)
      .def("clear", &NNCache::clear)
      .def_readonly("hits", &NNCache::hits)
      .def_readonly("lookups", &NNCache::lookups);

  py::class_<Search>(m, "Search")
      .def(py::init<SearchParams, std::shared_ptr<NNCache>>(), py::arg("params"), py::arg("cache") = nullptr)
      .def("set_position", &Search::set_position)
      .def("position", [](const Search& s) { return Position(s.position()); })
      .def("advance", &Search::advance)
      .def("begin", &Search::begin, py::arg("add_noise") = false)
      .def("done", &Search::done)
      .def("pending", &Search::pending)
      .def("root_visits", &Search::root_visits)
      .def("root_q", &Search::root_q)
      .def("root_d", &Search::root_d)
      .def("root_children",
           [](const Search& s) {
             py::list out;
             for (const auto& c : s.root_children()) out.append(py::make_tuple(c.move, c.policy_index, c.n, c.q, c.prior));
             return out;
           })
      .def("pv", &Search::pv, py::arg("max_len") = 32)
      .def("tree_size", &Search::tree_size)
      .def("max_depth", &Search::max_depth)
      .def("mean_depth", &Search::mean_depth);

  m.def(
      "gather_batch",
      [](const std::vector<Search*>& searches, int max_leaves, int target_visits) {
        Batch batch;
        std::vector<int32_t> per(searches.size());
        int not_done = 0;
        {
          py::gil_scoped_release release;
          for (size_t i = 0; i < searches.size(); ++i) {
            per[i] = searches[i]->gather(batch, max_leaves, target_visits);
            if (!searches[i]->done(target_visits)) ++not_done;
          }
        }
        const py::ssize_t rows = batch.rows;
        return py::make_tuple(to_array(batch.bb, {rows, 12}), to_array(batch.meta, {rows, 4}),
                              to_array(batch.legal, {py::ssize_t(batch.legal.size())}), to_array(batch.n_legal, {rows}),
                              to_array(per, {py::ssize_t(per.size())}), not_done);
      },
      py::arg("searches"), py::arg("max_leaves"), py::arg("target_visits"),
      "Collect leaves from every search. Returns (bb int64[B,12], meta uint8[B,4], legal_flat int64[L], "
      "n_legal int32[B], leaves_per_search int32[S], searches_not_done).");

  m.def(
      "apply_batch",
      [](const std::vector<Search*>& searches, py::array_t<float, py::array::c_style | py::array::forcecast> legal_logits,
         py::array_t<float, py::array::c_style | py::array::forcecast> wdl) {
        const float* lg = legal_logits.data();
        const float* w = wdl.data();
        py::ssize_t total_rows = 0, total_legal = 0;
        for (auto* s : searches) total_rows += s->pending(), total_legal += s->pending_legal();
        if (wdl.size() != total_rows * 3) throw std::invalid_argument("wdl has wrong number of rows");
        if (legal_logits.size() != total_legal) throw std::invalid_argument("legal_logits size mismatch");
        py::ssize_t off = 0, row = 0;
        {
          py::gil_scoped_release release;
          for (auto* s : searches) {
            const int k = s->pending();
            int consumed = 0;
            s->apply(lg + off, w + 3 * row, &consumed);
            off += consumed;
            row += k;
          }
        }
      },
      py::arg("searches"), py::arg("legal_logits"), py::arg("wdl"));
}
