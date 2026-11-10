# Origin

**AutoSoft's research prototypes for detecting GPUs that silently compute wrong answers.**

A GPU can be online, cool and report zero errors while some of its results are wrong. Nothing crashes, so the bad numbers end up inside your training run. This repo holds the first experiments in catching those faults from the renter's side, using only the access anyone renting a GPU already has.

> **Status:** research prototype. This is not production software yet.

---

## What's here

| File | What it does |
|---|---|
| `sdc_demo.py` | Multiplies two matrices on the GPU, flips one bit in the result to simulate a hardware fault, and catches it with a cheap checksum |

---

## Quick start

Any machine with an NVIDIA GPU and PyTorch works. A free Google Colab GPU is the easiest option.

```bash
pip install torch
python sdc_demo.py
```

**On Colab:** Runtime → Change runtime type → GPU, upload `sdc_demo.py`, then run `!python sdc_demo.py`.

---

## Results

Run on a Google Colab **Tesla T4**, using 2048 × 2048 FP32 matrices:

```
GPU: Tesla T4

Clean multiply   -> corrupted columns detected: []
Injected fault   -> C[1234,567] changed -53.6947 -> -3518934.7500 (no crash, no error)
Faulty multiply  -> corrupted columns detected: [567]

Multiply: 4.02 ms | Check: 0.36 ms | Overhead: 9.0%
```

- **One bit, a huge error.** Flipping bit 27, which sits in the exponent, multiplied the value by 2¹⁶ = 65,536. The program kept running normally.
- **Exact location.** The check pointed to column 567, which is where the fault was injected.
- **Cheap.** The check took 0.36 ms against 4.02 ms for the multiply.

Full write-up: [One flipped bit turned −53 into −3.5 million](https://autosoft-two.vercel.app/blog/one-flipped-bit.html)

---

## How it works

The check uses **algorithm-based fault tolerance** (Huang & Abraham, 1984). If `C = A · B`, then the column sums of `C` must equal the column sums of `A` multiplied by `B`:

```python
expected = A.sum(dim=0) @ B   # vector × matrix: O(n²)
actual   = C.sum(dim=0)       # column sums:     O(n²)
bad_cols = where(|expected - actual| / |expected| > tolerance)
```

The multiply costs O(n³), while the check costs O(n²). Floating-point rounding means the two sides never match exactly, so the check uses a relative tolerance of 1%.

**About the 9% overhead:** counting operations alone, the check should cost about 2/n of the multiply, roughly 0.1% at n = 2048. The measured 9% is mostly memory traffic and separate kernel launches, because the check is written as plain PyTorch calls. Building the checksum into the matrix-multiply kernel itself is the obvious next step.

---

## Limitations

To be clear about what this does **not** show yet:

- **The fault is hand-placed.** Real hardware faults come from specific defective units, can corrupt many values in a pattern, and may only appear under certain heat or voltage conditions.
- **Only one operation is checked.** Attention, normalisation, optimizer steps and communication between GPUs each need their own checks.
- **The tolerance is fixed.** A 1% threshold catches exponent flips but misses small flips in the low mantissa bits. Choosing thresholds well is especially hard for BF16 and FP8.
- **It ran on one healthy GPU.**

---

## Roadmap

Planned work, in rough order:

- [ ] Inject faults that match patterns published in recent studies of defective GPUs
- [ ] Add more detector types: known-answer tests, deterministic replay, memory pattern tests, cross-device consensus
- [ ] Score each detector on coverage, overhead and false-alarm rate
- [ ] Build the checksum into the matrix-multiply kernel to cut its overhead
- [ ] Run short screening tests across many rented GPUs from different providers
- [ ] Test whether the faults cheap checks miss actually change training results

This work belongs to AutoSoft's three product stages:

| Stage | Question | Status |
|---|---|---|
| **Origin** | Did my job stop? | prototype |
| **Frontier** | Can it keep going? | research |
| **Zenith** | Was the answer right? | active research |

More about the stages: [Origin, Frontier, Zenith](https://autosoft-two.vercel.app/blog/origin-frontier-zenith.html)

---

## Background reading

- K.-H. Huang and J. A. Abraham, "Algorithm-Based Fault Tolerance for Matrix Operations," *IEEE Transactions on Computers*, 1984
- Meta, [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783), 2024. See Table 5 for the breakdown of hardware interruptions.
- [AEGIS: Online SDC Detection and Insights from 35 Million GPU Hours](https://www.usenix.org/conference/osdi26/presentation/lei), OSDI 2026
- [LLM-PRISM: Silent Data Corruption from Permanent GPU Faults in LLM Training](https://arxiv.org/abs/2604.10390), 2026

---

## Contact

If you rent GPUs and have lost a run to hardware problems you couldn't explain, or you run GPU infrastructure, we'd like to hear from you.

**Website:** [autosoft-two.vercel.app](https://autosoft-two.vercel.app) · **X:** [@autosoft_inc](https://x.com/autosoft_inc)
