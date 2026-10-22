"""Concurrent match runner.

Up to ``concurrency`` games run at once. Each iteration, async players
(Stockfish) are asked for their moves first so they think in background
threads while batched players (our network) search all of their games in a
single batched MCTS run; then all chosen moves are applied.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import chess
import chess.pgn

from azchess.eval.openings import builtin_openings, opening_name
from azchess.game import CHECKMATE, TERMINAL_NAMES, make_position


@dataclass
class GameRecord:
    white: str
    black: str
    result: str  # "1-0", "0-1", "1/2-1/2"
    termination: str
    opening: str
    moves: list[str] = field(default_factory=list)
    start_fen: str = chess.STARTING_FEN
    infos: list[dict] = field(default_factory=list)

    def score_for(self, name: str) -> float:
        if self.result == "1/2-1/2":
            return 0.5
        white_won = self.result == "1-0"
        return float(white_won == (self.white == name))

    def to_pgn(self, event: str = "azchess match", round_: int | None = None) -> chess.pgn.Game:
        g = chess.pgn.Game()
        g.headers.update(
            Event=event, White=self.white, Black=self.black, Result=self.result, Termination=self.termination
        )
        if round_ is not None:
            g.headers["Round"] = str(round_)
        node = g
        board = chess.Board(self.start_fen)
        for i, uci in enumerate(self.moves):
            mv = chess.Move.from_uci(uci)
            node = node.add_variation(mv)
            info = self.infos[i] if i < len(self.infos) else {}
            if "q" in info:
                node.comment = f"q={info['q']:+.3f}" + (f" n={info['visits']}" if "visits" in info else "")
            board.push(mv)
        return g


def _result(code: int, white_to_move: bool) -> str:
    if code == CHECKMATE:
        return "0-1" if white_to_move else "1-0"
    return "1/2-1/2"


def play_match(
    player_a,
    player_b,
    n_games: int,
    openings: list[tuple[str, ...]] | None = None,
    concurrency: int = 16,
    max_plies: int = 400,
    verbose: bool = True,
    pgn_path: str | Path | None = None,
    record_infos: bool = False,
) -> list[GameRecord]:
    """Play ``n_games`` between two players, alternating colours on each opening."""
    openings = openings or builtin_openings()
    schedule = []
    for i in range(n_games):
        op = openings[(i // 2) % len(openings)]
        white, black = (player_a, player_b) if i % 2 == 0 else (player_b, player_a)
        schedule.append((i, white, black, op))
    queue = list(reversed(schedule))
    active: dict[int, dict] = {}
    records: list[GameRecord | None] = [None] * n_games
    done = 0
    t0 = time.time()
    tally: Counter = Counter()

    def start(i, white, black, op) -> None:
        pos = make_position(chess.STARTING_FEN)
        for m in op:
            pos.push_uci(m)
        for p in {id(white): white, id(black): black}.values():
            p.new_game(i, chess.STARTING_FEN, list(op))
        active[i] = {"pos": pos, "white": white, "black": black, "op": op, "moves": list(op), "infos": [{}] * len(op)}

    def finish(i: int, result: str, termination: str) -> None:
        nonlocal done
        g = active.pop(i)
        for p in {id(g["white"]): g["white"], id(g["black"]): g["black"]}.values():
            p.end_game(i)
        rec = GameRecord(
            g["white"].name,
            g["black"].name,
            result,
            termination,
            opening_name(g["op"]),
            g["moves"],
            infos=g["infos"] if record_infos else [],
        )
        records[i] = rec
        done += 1
        tally[rec.score_for(player_a.name)] += 1
        if verbose and (done % max(1, n_games // 10) == 0 or done == n_games):
            w, d, l = tally[1.0], tally[0.5], tally[0.0]
            print(
                f"[match] {player_a.name} vs {player_b.name}: {done}/{n_games}  +{w} ={d} -{l}  ({time.time() - t0:.0f}s)",
                flush=True,
            )

    while queue or active:
        while queue and len(active) < concurrency:
            start(*queue.pop())
        # Ask every player for the games where it is to move.
        by_player: dict[int, tuple[object, list[int]]] = {}
        for i, g in active.items():
            p = g["white"] if g["pos"].turn else g["black"]
            by_player.setdefault(id(p), (p, []))[1].append(i)
        futures = {}
        for p, gids in by_player.values():
            if not p.batched:
                for i in gids:
                    futures[i] = p.select_async(i)
        chosen: dict[int, tuple[str, dict]] = {}
        for p, gids in by_player.values():
            if p.batched:
                chosen.update(zip(gids, p.select(gids)))
        for i, fut in futures.items():
            chosen[i] = fut.result()
        for i, (uci, info) in chosen.items():
            g = active[i]
            g["pos"].push_uci(uci)
            g["moves"].append(uci)
            g["infos"] = g["infos"] + [info]
            for p in {id(g["white"]): g["white"], id(g["black"]): g["black"]}.values():
                p.observe(i, uci)
            code = g["pos"].terminal()
            if code:
                finish(i, _result(code, g["pos"].turn), TERMINAL_NAMES[code])
            elif len(g["moves"]) >= max_plies:
                finish(i, "1/2-1/2", "max plies")
    out = [r for r in records if r is not None]
    if pgn_path:
        Path(pgn_path).parent.mkdir(parents=True, exist_ok=True)
        with open(pgn_path, "a") as f:
            for k, r in enumerate(out):
                print(r.to_pgn(round_=k + 1), file=f, end="\n\n")
    return out


def summarize(records: list[GameRecord], name: str) -> dict:
    w = sum(r.score_for(name) == 1.0 for r in records)
    d = sum(r.score_for(name) == 0.5 for r in records)
    l = len(records) - w - d
    return {"wins": w, "draws": d, "losses": l, "games": len(records), "score": (w + d / 2) / max(len(records), 1)}


def records_to_json(records: list[GameRecord]) -> list[dict]:
    return [{k: v for k, v in asdict(r).items() if k != "infos"} for r in records]
