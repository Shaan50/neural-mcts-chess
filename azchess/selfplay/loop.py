"""Self-play reinforcement loop with gating (AlphaGo Zero style).

Each generation:

1. **self-play**: the current *best* network plays ``selfplay.games`` games;
2. **train**: a candidate continues training from the previous candidate on a
   replay buffer of the last ``window`` generations of self-play data, mixed
   with a fraction ``sl_mix`` of human-game batches (prevents catastrophic
   forgetting while self-play data is scarce);
3. **gate**: candidate vs best over ``gating_games`` games with paired
   openings; the candidate becomes the new best if it scores at least
   ``gating_threshold``. An SPRT verdict and Elo difference (95% CI) are logged;
4. optionally **evaluate** the best network against a Stockfish ladder to
   track absolute Elo per generation.

State is checkpointed in ``state.json`` after every phase so the loop can be
interrupted and resumed.

    python -m azchess.selfplay.loop --config configs/selfplay.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml

from azchess.data.loader import make_loader
from azchess.data.shards import list_shards, shard_stats
from azchess.eval.elo import performance, sprt
from azchess.eval.match import play_match, summarize
from azchess.eval.players import MCTSPlayer
from azchess.eval.run import evaluate
from azchess.mcts.pysearch import SearchParams
from azchess.model.inference import Evaluator
from azchess.selfplay.selfplay import SelfPlayConfig, generate
from azchess.train.trainer import TrainConfig, Trainer


@dataclass
class LoopConfig:
    run_dir: str = "runs/selfplay"
    init_checkpoint: str = "runs/sl_medium/best.pt"
    generations: int = 5
    selfplay: dict = field(default_factory=dict)
    window: int = 3
    sl_data: list[str] = field(default_factory=lambda: ["data/shards/train"])
    sl_mix: float = 0.3
    val_data: list[str] = field(default_factory=lambda: ["data/shards/val"])
    train_steps: int = 600
    batch_size: int = 512
    lr: float = 3e-4
    weight_decay: float = 0.01
    q_ratio: float = 0.25
    value_weight: float = 1.0
    gating_games: int = 100
    gating_sims: int = 200
    gating_threshold: float = 0.55
    eval_opponents: list[str] = field(default_factory=list)
    eval_games: int = 40
    eval_sims: int = 200
    device: str = "auto"
    seed: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> LoopConfig:
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown loop config keys: {sorted(unknown)}")
        return cls(**d)


def mixed_batches(sp_loader, sl_loader, p_sl: float, seed: int):
    rng = np.random.default_rng(seed)
    it_sp = iter(sp_loader)
    it_sl = iter(sl_loader) if sl_loader is not None else None
    while True:
        if it_sl is not None and rng.random() < p_sl:
            yield next(it_sl)
        else:
            yield next(it_sp)


class Loop:
    def __init__(self, cfg: LoopConfig):
        self.cfg = cfg
        self.dir = Path(cfg.run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "state.json"
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
            print(f"[loop] resuming at generation {self.state['generation']} (best={self.state['best']})")
        else:
            init = self.dir / "gen_000_best.pt"
            shutil.copy(cfg.init_checkpoint, init)
            self.state = {
                "generation": 1,
                "phase": "selfplay",
                "best": str(init),
                "best_gen": 0,
                "latest": str(init),
                "elo": {"0": 0.0},
                "history": [],
            }
            self._save()
        (self.dir / "config.yaml").write_text(yaml.safe_dump(asdict(cfg)))

    def _save(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2))
        tmp.replace(self.state_path)

    def gen_dir(self, g: int) -> Path:
        return self.dir / f"gen_{g:03d}"

    # ----------------------------------------------------------- phases
    def selfplay(self, g: int) -> dict:
        sp = SelfPlayConfig.from_dict({**self.cfg.selfplay, "seed": self.cfg.seed * 1000 + g})
        ev = Evaluator.from_checkpoint(self.state["best"], self.cfg.device)
        stats = generate(ev, sp, self.gen_dir(g) / "selfplay", prefix=f"sp{g:03d}")
        return stats.summary()

    def train(self, g: int) -> dict:
        cfg = self.cfg
        shards = []
        for k in range(max(1, g - cfg.window + 1), g + 1):
            shards += list_shards(self.gen_dir(k) / "selfplay")
        tc = TrainConfig(
            run_dir=str(self.gen_dir(g) / "train"),
            train_data=shards,
            val_data=cfg.val_data,
            model={},
            batch_size=cfg.batch_size,
            steps=cfg.train_steps,
            lr=cfg.lr,
            min_lr_ratio=0.1,
            warmup_steps=min(100, cfg.train_steps // 10),
            weight_decay=cfg.weight_decay,
            value_weight=cfg.value_weight,
            q_ratio=cfg.q_ratio,
            ema_decay=0.99,
            workers=2,
            log_every=50,
            val_every=cfg.train_steps,
            val_positions=100_000,
            ckpt_every=cfg.train_steps,
            snapshot_every=cfg.train_steps,
            seed=cfg.seed + g,
            device=cfg.device,
        )
        from azchess.model.inference import load_checkpoint

        model, _ = load_checkpoint(self.state["latest"], "cpu")
        trainer = Trainer(tc, model=model)
        sp_loader = make_loader(shards, cfg.batch_size, workers=2, seed=cfg.seed + g, shards_in_memory=8)
        sl_loader = (
            make_loader(cfg.sl_data, cfg.batch_size, workers=2, seed=cfg.seed + 7 * g) if cfg.sl_mix > 0 else None
        )

        def val_factory():
            return make_loader(
                cfg.val_data, 2048, workers=0, shuffle=False, infinite=False, max_positions=tc.val_positions
            )

        try:
            trainer.fit(mixed_batches(sp_loader, sl_loader, cfg.sl_mix, cfg.seed + g), val_factory)
        finally:
            trainer.close()
        cand = self.gen_dir(g) / "candidate.pt"
        shutil.copy(Path(tc.run_dir) / "final.pt", cand)
        self.state["latest"] = str(cand)
        metrics = [json.loads(line) for line in open(Path(tc.run_dir) / "metrics.jsonl")]
        val = [m for m in metrics if m["split"] == "val"]
        return {"replay_positions": shard_stats(shards)["positions"], "val": val[-1] if val else None}

    def gate(self, g: int) -> dict:
        cfg = self.cfg
        cand_ev = Evaluator.from_checkpoint(self.state["latest"], cfg.device)
        best_ev = Evaluator.from_checkpoint(self.state["best"], cfg.device)
        params = SearchParams()
        cand = MCTSPlayer(cand_ev, cfg.gating_sims, params, name=f"gen{g}")
        best = MCTSPlayer(best_ev, cfg.gating_sims, params, name=f"best_gen{self.state['best_gen']}")
        recs = play_match(
            cand, best, cfg.gating_games, concurrency=min(64, cfg.gating_games), pgn_path=self.gen_dir(g) / "gating.pgn"
        )
        s = summarize(recs, cand.name)
        perf = performance(s["wins"], s["draws"], s["losses"])
        test = sprt(s["wins"], s["draws"], s["losses"], elo0=0, elo1=30)
        accepted = s["score"] >= cfg.gating_threshold
        out = {
            **s,
            "elo_diff": perf.elo,
            "elo_lo": perf.lo,
            "elo_hi": perf.hi,
            "sprt_llr": test.llr,
            "sprt": test.decision,
            "accepted": accepted,
        }
        print(
            f"[loop] gen {g} gating: {s} elo {perf} sprt {test.decision} -> {'ACCEPT' if accepted else 'reject'}",
            flush=True,
        )
        if accepted:
            new_best = self.dir / f"gen_{g:03d}_best.pt"
            shutil.copy(self.state["latest"], new_best)
            self.state["elo"][str(g)] = self.state["elo"][str(self.state["best_gen"])] + perf.elo
            self.state["best"], self.state["best_gen"] = str(new_best), g
        return out

    def stockfish_eval(self, g: int) -> dict | None:
        cfg = self.cfg
        if not cfg.eval_opponents:
            return None
        res = evaluate(
            f"nn:path={self.state['best']},sims={cfg.eval_sims}",
            cfg.eval_opponents,
            cfg.eval_games,
            device=cfg.device,
            bootstrap=200,
        )
        (self.gen_dir(g) / "sf_eval.json").write_text(json.dumps(res, indent=2))
        return res.get("rating")

    def run(self) -> None:
        cfg = self.cfg
        while self.state["generation"] <= cfg.generations:
            g = self.state["generation"]
            entry = self.state.setdefault("current", {"generation": g})
            t0 = time.time()
            if self.state["phase"] == "selfplay":
                entry["selfplay"] = self.selfplay(g)
                self.state["phase"] = "train"
                self._save()
            if self.state["phase"] == "train":
                entry["train"] = self.train(g)
                self.state["phase"] = "gate"
                self._save()
            if self.state["phase"] == "gate":
                entry["gating"] = self.gate(g)
                self.state["phase"] = "eval"
                self._save()
            if self.state["phase"] == "eval":
                entry["stockfish"] = self.stockfish_eval(g)
                entry["best_gen"] = self.state["best_gen"]
                entry["seconds"] = time.time() - t0
                self.state["history"].append(entry)
                self.state.pop("current")
                self.state["generation"] = g + 1
                self.state["phase"] = "selfplay"
                self._save()
        print(f"[loop] finished; best = {self.state['best']} (generation {self.state['best_gen']})")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("overrides", nargs="*")
    a = p.parse_args(argv)
    d = yaml.safe_load(open(a.config))
    from azchess.train.supervised import parse_overrides

    for k, v in parse_overrides(a.overrides).items():
        if k.startswith("selfplay."):
            d.setdefault("selfplay", {})[k[9:]] = v
        else:
            d[k] = v
    Loop(LoopConfig.from_dict(d)).run()


if __name__ == "__main__":
    main()
