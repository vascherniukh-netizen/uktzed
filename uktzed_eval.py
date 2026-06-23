# -*- coding: utf-8 -*-
"""
Eval-харнес ретривера УКТ ЗЕД.

Міряє те, що реально визначає помилки агента: чи правильний код узагалі
потрапляє в кандидати (recall) і на якому місці.

Метрики на трьох рівнях деталізації:
  exact (10 знаків) — точний код
  sub   (6 знаків)  — субпозиція (правильна «гілка»)
  pos   (4 знаки)   — товарна позиція (правильна «околиця»)

Для кожного рівня:
  recall@k  — частка кейсів, де еталон є серед топ-k кандидатів
  top1      — частка, де еталон на 1-му місці
  top5      — частка, де еталон у топ-5
  MRR       — середній обернений ранг

Провали розкладаються на кошики:
  RECALL_MISS — еталона немає в топ-k (найгірше: агент його не побачить)
  RANK_MISS   — еталон є, але не на 1-му місці (вирішує уточнення/реранк)

Запуск:
  python uktzed_eval.py                  # сід-набір, k=8, лексичний ретривер
  python uktzed_eval.py --k 20           # підняти top-k
  python uktzed_eval.py --dense          # увімкнути e5 (локально, потрібна модель)
  python uktzed_eval.py --data mine.jsonl --show-fails 30
"""
import os, sys, json, argparse
from uktzed_retriever import UktzedRetriever

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


def first_rank(codes, gold, width):
    """1-індексований ранг першого кандидата, що збігається з gold на перших `width` знаках; інакше None."""
    g = gold[:width]
    for i, c in enumerate(codes):
        if c[:width] == g:
            return i + 1
    return None


def evaluate(retriever, cases, k):
    levels = {"exact": 10, "sub": 6, "pos": 4}
    agg = {lv: {"recall": 0, "top1": 0, "top5": 0, "mrr": 0.0} for lv in levels}
    rows = []
    for case in cases:
        hits = retriever.search(case["desc"], top_k=k)
        codes = [h["code"] for h in hits]
        gold = case["gold"].replace(" ", "")
        ranks = {}
        for lv, w in levels.items():
            r = first_rank(codes, gold, w)
            ranks[lv] = r
            if r is not None:
                agg[lv]["recall"] += 1
                agg[lv]["mrr"] += 1.0 / r
                if r == 1: agg[lv]["top1"] += 1
                if r <= 5: agg[lv]["top5"] += 1
        # кошик провалу за exact
        if ranks["exact"] is None:
            bucket = "RECALL_MISS"
        elif ranks["exact"] > 1:
            bucket = "RANK_MISS"
        else:
            bucket = "OK"
        rows.append({"desc": case["desc"], "gold": gold,
                     "top1": codes[0] if codes else "—",
                     "rank": ranks["exact"], "pos_rank": ranks["pos"],
                     "bucket": bucket})
    n = len(cases)
    for lv in levels:
        for m in ("recall", "top1", "top5"):
            agg[lv][m] = agg[lv][m] / n
        agg[lv]["mrr"] = agg[lv]["mrr"] / n
    return agg, rows, n


def report(agg, rows, n, k):
    print(f"\nКейсів: {n}   top-k: {k}\n")
    print(f"{'рівень':<7} {'recall@k':>9} {'top-1':>7} {'top-5':>7} {'MRR':>6}")
    print("-" * 40)
    names = {"exact": "exact10", "sub": "sub6", "pos": "pos4"}
    for lv in ("exact", "sub", "pos"):
        a = agg[lv]
        print(f"{names[lv]:<7} {a['recall']*100:>8.1f}% {a['top1']*100:>6.1f}% "
              f"{a['top5']*100:>6.1f}% {a['mrr']:>6.3f}")

    buckets = {"RECALL_MISS": [], "RANK_MISS": [], "OK": []}
    for r in rows:
        buckets[r["bucket"]].append(r)
    print(f"\nКошики:  OK={len(buckets['OK'])}  "
          f"RANK_MISS={len(buckets['RANK_MISS'])}  "
          f"RECALL_MISS={len(buckets['RECALL_MISS'])}")

    if buckets["RECALL_MISS"]:
        print("\n⚠ RECALL_MISS — еталон не потрапив у топ-k (агент його не побачить):")
        for r in buckets["RECALL_MISS"]:
            pos = f"pos4 ранг {r['pos_rank']}" if r["pos_rank"] else "навіть позиція мимо"
            print(f"   «{r['desc']}»\n      еталон {r['gold']} · топ-1 був {r['top1']} · {pos}")
    if buckets["RANK_MISS"]:
        print("\n· RANK_MISS — еталон у топ-k, але не №1 (лікується уточненням/реранком):")
        for r in buckets["RANK_MISS"]:
            print(f"   «{r['desc']}» · еталон {r['gold']} на місці {r['rank']} · топ-1 {r['top1']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__) or ".", "uktzed_eval_seed.jsonl"))
    ap.add_argument("--jsonl", default=os.path.join(os.path.dirname(__file__) or ".", "uktzed_nodes.jsonl"))
    ap.add_argument("--dense", action="store_true")
    ap.add_argument("--show-fails", type=int, default=999)
    a = ap.parse_args()

    cases = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    r = UktzedRetriever(a.jsonl, use_dense=a.dense)
    agg, rows, n = evaluate(r, cases, a.k)
    report(agg, rows, n, a.k)


if __name__ == "__main__":
    main()
