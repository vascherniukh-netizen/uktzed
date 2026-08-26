# -*- coding: utf-8 -*-
"""
Підбір ваг RRF-злиття на eval-сіді. Рахує скори кожного ранкера ОДИН раз,
далі перебирає сітку ваг за секунди (без повторних пошуків).

Захист від самообману: --holdout ділить сід навпіл детерміновано (за хешем gold):
тюнимо на половині A, звітуємо на half B. Ваги, що виграють лише на A — перепідгонка.

Запуск:
    python uktzed_tune.py                # лексика (bm25 + char)
    python uktzed_tune.py --dense        # + e5 (потрібна модель)
    python uktzed_tune.py --dense --holdout
"""
import os, sys, json, argparse, hashlib
import numpy as np
from uktzed_retriever import UktzedRetriever, tokenize

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

TOP_N = 200   # скільки позицій кожен ранкер нагороджує в RRF (як у ретривері)
K_RRF = 60
K_EVAL = 20


def precompute(r, cases):
    """Для кожного кейсу: (indices, scores) топ-TOP_N кожного ранкера."""
    per_case = []
    for c in cases:
        arrays = [r.bm25.scores(tokenize(c['desc'])), r.char.scores(c['desc'])]
        if r.dense is not None:
            arrays.append(r.dense.scores(c['desc']))
        ranked = []
        for s in arrays:
            idx = np.argsort(-s)[:TOP_N]
            idx = idx[s[idx] > 0]
            ranked.append((idx, s[idx]))
        per_case.append(ranked)
    return per_case


def fuse_metrics(per_case, cases, codes, weights):
    rec = t1 = t5 = 0; mrr = 0.0
    for ranked, c in zip(per_case, cases):
        acc = {}
        for w, (idx, _) in zip(weights, ranked):
            for rank, i in enumerate(idx):
                acc[i] = acc.get(i, 0.0) + w / (K_RRF + rank + 1)
        top = sorted(acc, key=acc.get, reverse=True)[:K_EVAL]
        gold = c['gold']
        hit = next((p for p, i in enumerate(top, 1) if codes[i] == gold), None)
        if hit:
            rec += 1; mrr += 1.0 / hit
            t1 += (hit == 1); t5 += (hit <= 5)
    n = len(cases)
    return rec/n, t1/n, t5/n, mrr/n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dense', action='store_true')
    ap.add_argument('--holdout', action='store_true')
    ap.add_argument('--data', default='uktzed_eval_seed.jsonl')
    ap.add_argument('--jsonl', default='uktzed_nodes.jsonl')
    a = ap.parse_args()

    cases = [json.loads(l) for l in open(a.data, encoding='utf-8') if l.strip()]
    r = UktzedRetriever(a.jsonl, use_dense=a.dense)
    codes = [n['code'] for n in r.leaves]
    names = ['bm25', 'char'] + (['e5'] if r.dense is not None else [])

    print(f"Рахую скори {len(names)} ранкерів на {len(cases)} кейсах (раз)…")
    per_case = precompute(r, cases)

    if a.holdout:
        split = lambda c: int(hashlib.md5(c['gold'].encode()).hexdigest(), 16) % 2
        A = [i for i, c in enumerate(cases) if split(c) == 0]
        B = [i for i, c in enumerate(cases) if split(c) == 1]
        print(f"holdout: тюн на {len(A)}, звіт на {len(B)}")
    else:
        A = B = list(range(len(cases)))

    grid = [0.2, 0.4, 0.7, 1.0, 1.3]
    import itertools
    combos = list(itertools.product(grid, repeat=len(names)))
    scored = []
    pcA = [per_case[i] for i in A]; csA = [cases[i] for i in A]
    for w in combos:
        rec, t1, t5, mrr = fuse_metrics(pcA, csA, codes, w)
        scored.append((rec, mrr, w))
    scored.sort(reverse=True)

    print(f"\nТоп-8 на {'половині A' if a.holdout else 'всьому сіді'} "
          f"(ваги: {'/'.join(names)}):")
    print(f"{'ваги':<18}{'recall@20':>10}{'top-1':>8}{'top-5':>8}{'MRR':>8}")
    pcB = [per_case[i] for i in B]; csB = [cases[i] for i in B]
    for rec, mrr, w in scored[:8]:
        recB, t1B, t5B, mrrB = fuse_metrics(pcB, csB, codes, w)
        tag = '' if not a.holdout else f"   | B: {recB*100:.0f}%/{mrrB:.3f}"
        print(f"{str(w):<18}{rec*100:>9.1f}%{'':>0}{'':>0}{mrr:>24.3f}{tag}"
              if False else
              f"{str(w):<18}{rec*100:>9.1f}% {fuse_metrics(pcA,csA,codes,w)[1]*100:>6.1f}% "
              f"{fuse_metrics(pcA,csA,codes,w)[2]*100:>6.1f}% {mrr:>7.3f}{tag}")

    # поточні дефолти для порівняння
    defaults = (0.7, 0.4, 1.3)[:len(names)] if a.dense else (0.5, 1.3)
    rec, t1, t5, mrr = fuse_metrics(per_case, cases, codes, defaults)
    print(f"\nПоточні дефолти {defaults}: recall@20 {rec*100:.1f}%, top-1 {t1*100:.1f}%, "
          f"top-5 {t5*100:.1f}%, MRR {mrr:.3f} (на всьому сіді)")

    # сольні ранкери — для розуміння, хто тягне
    print("\nСольні ранкери (на всьому сіді):")
    for i, nm in enumerate(names):
        w = [0.0]*len(names); w[i] = 1.0
        rec, t1, t5, mrr = fuse_metrics(per_case, cases, codes, w)
        print(f"  {nm:<6} recall@20 {rec*100:.1f}%  top-1 {t1*100:.1f}%  MRR {mrr:.3f}")

if __name__ == '__main__':
    main()
