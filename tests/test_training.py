import numpy as np
import torch

from neural_mcts_chess.encoding.board import planes_from_compact_torch
from neural_mcts_chess.encoding.moves import POLICY_SIZE
from neural_mcts_chess.game import PyPosition
from neural_mcts_chess.model.network import AZNet, ModelConfig, value_to_wdl
from neural_mcts_chess.train.losses import compute_losses, policy_targets, value_targets
from neural_mcts_chess.train.trainer import TrainConfig, Trainer


def _batch(n=8, seed=0):
    rng = np.random.default_rng(seed)
    pos = PyPosition()
    bbs, metas, flats, moves = [], [], [], []
    for i in range(n):
        bb, meta = pos.encode()
        legal = pos.legal_moves()
        idx = pos.policy_indices(legal)
        k = int(rng.integers(len(legal)))
        bbs.append(bb)
        metas.append(meta)
        flats.append(i * POLICY_SIZE + idx)
        moves.append(idx[k])
        pos.push(legal[k])
    return {
        "bb": torch.from_numpy(np.stack(bbs).view(np.int64)),
        "meta": torch.from_numpy(np.stack(metas)),
        "policy_idx": torch.tensor(moves)[:, None],
        "policy_p": torch.ones(n, 1),
        "result": torch.from_numpy(rng.integers(-1, 2, n)),
        "q": torch.full((n,), float("nan")),
        "legal_flat": torch.from_numpy(np.concatenate(flats)),
    }


def test_network_shapes_and_param_count():
    for vh in ("wdl", "scalar"):
        m = AZNet(ModelConfig(blocks=2, filters=16, value_head=vh))
        p, v = m(torch.zeros(3, 22, 8, 8))
        assert p.shape == (3, POLICY_SIZE)
        assert v.shape == (3, 3 if vh == "wdl" else 1)
        wdl = value_to_wdl(v, vh)
        assert torch.allclose(wdl.sum(-1), torch.ones(3), atol=1e-5)


def test_policy_targets_handle_padding():
    idx = torch.tensor([[5, 7, -1], [3, -1, -1]])
    p = torch.tensor([[0.25, 0.75, 0.0], [1.0, 0.0, 0.0]])
    t = policy_targets(idx, p)
    assert torch.allclose(t.sum(-1), torch.ones(2))
    assert t[0, 7] == 0.75 and t[1, 3] == 1.0


def test_value_targets_q_mixing():
    z = value_targets(torch.tensor([1, 0, -1]), torch.tensor([float("nan"), 0.5, -1.0]), 0.5)
    assert torch.allclose(z[0], torch.tensor([1.0, 0.0, 0.0]))  # unknown q -> pure result
    assert torch.allclose(z[1], torch.tensor([0.25, 0.75, 0.0]))
    assert torch.allclose(z[2], torch.tensor([0.0, 0.0, 1.0]))


def test_masked_loss_ignores_illegal_logits():
    b = _batch()
    logits = torch.zeros(8, POLICY_SIZE)
    illegal = torch.ones(8 * POLICY_SIZE, dtype=torch.bool)
    illegal[b["legal_flat"]] = False
    logits.view(-1)[illegal] = 100.0  # huge logits on illegal moves must not matter
    pl, vl, m = compute_losses(logits, torch.zeros(8, 3), b)
    n_legal = torch.bincount(b["legal_flat"] // POLICY_SIZE, minlength=8).float()
    assert torch.allclose(pl, torch.log(n_legal).mean(), atol=1e-4)
    assert torch.isfinite(vl)
    assert 0 <= float(m["policy_top1"]) <= 1


def test_overfit_single_batch():
    torch.manual_seed(0)
    b = _batch(16)
    m = AZNet(ModelConfig(blocks=1, filters=16))
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    x = planes_from_compact_torch(b["bb"], b["meta"])
    first = None
    for _ in range(150):
        p, v = m(x)
        pl, vl, metrics = compute_losses(p, v, b)
        loss = pl + vl
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.2 * first
    assert float(metrics["policy_top1"]) == 1.0


def test_trainer_checkpoint_roundtrip(tmp_path):
    cfg = TrainConfig(
        run_dir=str(tmp_path / "run"),
        model={"blocks": 1, "filters": 8},
        steps=3,
        device="cpu",
        amp=False,
        log_every=1,
        ckpt_every=3,
        snapshot_every=3,
        warmup_steps=1,
    )
    t = Trainer(cfg)
    batches = iter([_batch(4, s) for s in range(10)])
    t.fit(batches)
    t.close()
    assert (tmp_path / "run" / "last.pt").exists() and (tmp_path / "run" / "snapshots" / "step_0000003.pt").exists()
    t2 = Trainer(TrainConfig(**{**cfg.__dict__, "steps": 5}))
    assert t2.step == 3
    for a, b in zip(t.model.state_dict().values(), t2.model.state_dict().values()):
        assert torch.equal(a.cpu(), b.cpu())
    t2.close()
