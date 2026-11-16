<img src="../assets/figures/header-phase2.svg" alt="Phase 2, the on-site final in Dublin, 14 to 15 Nov 2026: four live rounds, one pooled slate. Round 3 score 53.71, the highest of all 15 teams. 400,000 requests per round in rounds 3 and 4. 240 seconds per scoring run." width="100%">

<sub>[← Overview](../README.md) &nbsp;·&nbsp; [Phase 1](../phase1/) &nbsp;·&nbsp; <b>Phase 2</b> &nbsp;·&nbsp; [Round 3 →](round3/) &nbsp;·&nbsp; [Round 4 →](round4/)</sub>

# Phase 2: the on-site final

Fifteen teams, two days in Dublin, four live rounds. Each round gave a data folder and a list of
requests (one user at one moment). For each request every team recommended **one** item, all the
picks were pooled into one slate, and the simulated user chose from it. Credit for a request is the
chance the user picks your item, if they click anything.

<img src="../assets/figures/scoring.svg" alt="Each team recommends one item. Identical picks merge into one slate of items A, B and C. The simulated user would pick A 25%, B 55% and C 20% of the time if they click. Every team that picked B gets 55." width="100%">

## The two entries in this folder

<a href="round3/"><img src="../assets/figures/explore-round3.svg" alt="Round 3, top score of the round: the latest-click entry. Each user's most credible past click, in one readable file. No training. Opens phase2/round3/." width="100%"></a>

<a href="round4/"><img src="../assets/figures/explore-round4.svg" alt="Round 4, trained in the run: the ranker entry. LightGBM over each user's past clicks, trained inside the 240-second limit. Opens phase2/round4/." width="100%"></a>

Each folder is exactly one submission: zip its contents and it runs. Each README says what the entry
does, the measured evidence behind every setting, and how to run it on a laptop. To check both entries
without any data, run `python phase2/test_entries.py`: it builds a tiny made-up round and checks that
every request gets exactly one item that is available at its time.

## The four rounds

<img src="../assets/figures/rounds.svg" alt="Round scores. Round 1: NextToken 17.40, top 38.05. Round 2: 46.97, top 52.16. Round 3: 53.71, the top score of all teams. Round 4: 73.67, top 74.15." width="100%">

| Round | Data | What we submitted | Score |
| --- | --- | --- | ---: |
| 1 | First simulated world, 75,000 requests | The most-clicked item for almost everyone | 17.40 |
| 2 | Same world plus round 1's slates | Each user's best past click, plus two general items | 46.97 |
| 3 (×2) | **New** world: 20,000 users, 400,000 requests | [Each user's latest past click](round3/) | **53.71**, top score |
| 4 (×2) | Round 3's world plus its pooled slates | [A ranker trained during the scoring run](round4/) | **73.67** |

Rounds 1 and 2 taught the two lessons that set up day two (trust the direct test; fresh is not the
same as strong). They are told on the [front page](../README.md#rounds-1-and-2-two-lessons-that-set-up-day-two);
their code is not published here.

## The rules every entry had to meet

| | |
| --- | --- |
| Entry | `model.zip` with `solution.py`, whose `predict(data_dir, batch_size)` yields `(rowid, itemID)` pairs |
| Time | 240 seconds for the whole run, imports and training included; rows yielded before a timeout still count |
| Machine | No network, read-only disk, `/tmp` writable, only the preinstalled packages |
| Valid pick | An item from the catalogue that is available at the request time |
| AI tools | Allowed on the team's laptop only; the competition machine was operated by hand |

<img src="../assets/figures/divider.svg" alt="" width="100%">

<sub>[← Phase 1](../phase1/) &nbsp;·&nbsp; <b>Start with</b> [Round 3, the top score of the round →](round3/)</sub>
