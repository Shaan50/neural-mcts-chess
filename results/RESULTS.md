### Held-out move prediction

| Model | Params | Top-1 | Top-5 | Policy CE | Value MSE |
|---|---:|---:|---:|---:|---:|
| 4×32 | 0.36M | 43.9% | 82.3% | 1.813 | 0.743 |
| 6×64 | 0.76M | 49.2% | 87.7% | 1.580 | 0.728 |
| 10×128 | 3.40M | 52.7% | 90.2% | 1.441 | 0.714 |

Breakdown for the 10×128 network:

| Slice | Positions | Top-1 | Top-5 |
|---|---:|---:|---:|
| ply 0-20 | 108,987 | 54.4% | 92.6% |
| ply 20-40 | 103,228 | 51.0% | 88.9% |
| ply 40-60 | 82,311 | 51.0% | 88.2% |
| ply 60-80 | 51,476 | 52.6% | 89.5% |
| ply 80-1000 | 52,967 | 55.0% | 91.9% |
| rating 2000-2200 | 255,367 | 52.7% | 90.2% |
| rating 2200-2400 | 108,653 | 53.0% | 90.6% |
| rating 2400-2600 | 28,030 | 52.3% | 90.2% |
| rating 2600-4000 | 6,919 | 46.9% | 87.1% |

### Raw policy vs MCTS

| Search | Elo [95% CI] | Games (W/D/L) | Score by Stockfish level |
|---|---|---|---|
| raw policy (no search) | **1447** [1367, 1521] | +37 =10 −61 | 1320: 0.66, 1520: 0.36, 1720: 0.22 |
| MCTS, 16 visits/move | **1570** [1501, 1635] | +55 =5 −48 | 1320: 0.89, 1520: 0.62, 1720: 0.22 |
| MCTS, 64 visits/move | **1733** [1669, 1803] | +70 =9 −29 | 1350: 0.94, 1550: 0.77, 1750: 0.52 |
| MCTS, 256 visits/move | **2147** [2069, 2237] | +81 =8 −19 | 1700: 0.83, 1900: 0.80, 2100: 0.75 |
| MCTS, 800 visits/move | **2548** [2473, 2633] | +81 =13 −26 | 2200: 0.89, 2400: 0.66, 2600: 0.50 |

### Tactics (Win At Chess, 300 positions)

| Network | raw policy | 64 visits | 256 visits | 800 visits | 3200 visits |
|---|---:|---:|---:|---:|---:|
| medium | 50/300 | 70/300 | 103/300 | 118/300 | 131/300 |
| tiny | 31/300 | 38/300 | 49/300 | 67/300 | 78/300 |

### Network size vs strength

| Network | Params | Search | Elo [95% CI] | Games (W/D/L) |
|---|---:|---|---|---|
| tiny (4×32) | 0.36M | raw policy | **1043** [871, 1141] | +7 =2 −99 |
| small (6×64) | 0.76M | raw policy | **1409** [1341, 1476] | +31 =13 −64 |
| medium (10×128) | 3.40M | raw policy | **1465** [1390, 1532] | +31 =9 −68 |
| tiny (4×32) | 0.36M | MCTS 256 | **1613** [1551, 1675] | +50 =8 −50 |
| small (6×64) | 0.76M | MCTS 256 | **1914** [1828, 2002] | +67 =6 −35 |
| medium (10×128) | 3.40M | MCTS 256 | **2250** [2185, 2318] | +62 =12 −46 |

### Elo vs training step

| Training step | Positions | Search | Elo [95% CI] |
|---:|---:|---|---|
| 4,000 | 4.1M | raw policy | **1255** [1165, 1333] |
| 8,000 | 8.2M | raw policy | **1479** [1404, 1547] |
| 12,000 | 12.3M | raw policy | **1459** [1395, 1520] |
| 16,000 | 16.4M | raw policy | **1465** [1397, 1534] |
| 20,000 | 20.5M | raw policy | **1594** [1514, 1662] |
| 24,000 | 24.6M | raw policy | **1479** [1418, 1541] |
| 28,000 | 28.7M | raw policy | **1498** [1428, 1563] |
| 30,000 | 30.7M | raw policy | **1535** [1468, 1611] |

### Pretraining-only vs self-play-refined

| Network | Elo vs Stockfish [95% CI] | Games (W/D/L) |
|---|---|---|
| pretrained (MCTS 256) | **2157** [2075, 2241] | +71 =11 −26 |
| selfplay (MCTS 256) | **2292** [2218, 2364] | +64 =19 −37 |

Head-to-head, self-play-refined vs pretrained (300 games, paired openings): +90 =137 −73, score 0.528, Elo difference **+20** [-9, +49]

| Gen | Self-play games | Positions recorded | Gating score (vs current best) | Elo diff [95% CI] | SPRT | Accepted |
|---:|---:|---:|---:|---|---|---|
| 1 | 1,200 | 26,154 | 0.495 | -3 [-57, +50] | continue | no |
| 2 | 1,200 | 26,393 | 0.490 | -7 [-57, +43] | continue | no |
| 3 | 1,200 | 26,580 | 0.595 | +67 [+14, +117] | continue | yes |
| 4 | 1,200 | 24,644 | 0.460 | -28 [-81, +26] | continue | no |
| 5 | 1,200 | 24,505 | 0.520 | +14 [-42, +70] | continue | no |
| 6 | 1,200 | 24,832 | 0.525 | +17 [-36, +70] | continue | no |
| 7 | 1,200 | 24,593 | 0.475 | -17 [-73, +39] | continue | no |
| 8 | 1,200 | 24,538 | 0.485 | -10 [-63, +42] | continue | no |

