# -*- coding: utf-8 -*-
"""
Живий тест агента на реальному DeepSeek API. НЕ безкоштовний — кожен запуск
витрачає токени (~10-20 API-викликів). Ганяти вручну або через /live-test у Claude Code.

Що перевіряє, чого не бачить uktzed_eval.py:
  - чи модель переформульовує жаргон мовою тарифу (правило 2a) і робить повторний пошук;
  - чи фінальний код проходить finalize_code і збігається з очікуваним;
  - чи ставить рівно одне уточнююче питання там, де це доречно.

Запуск:
    python uktzed_live_test.py --dense        # бойовий прогін (потрібен DEEPSEEK_API_KEY)
    python uktzed_live_test.py --plumbing     # перевірка самого харнесу, без API

Вихідний код процесу: 0 — усі PASS, 1 — є FAIL.
"""
import sys, argparse, json
from uktzed_retriever import UktzedRetriever
from uktzed_agent import UktzedAgent, make_client, MODEL

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# expected — префікс коду (10 знаків = точний код). followup — відповідь на уточнююче
# питання моделі, якщо вона його поставить. jargon=True — очікуємо переформулювання.
CASES = [
    {"q": "підводний акустичний маяк-пінгер для позиціювання обʼєктів",
     "expected": "9014", "followup": "це навігаційний прилад, не рятувальний буй",
     "jargon": True,
     "note": "головний тест правила 2a: у тарифі немає жодного слова із запиту"},
    {"q": "болгарка 125 мм",
     "expected": "8467", "followup": "електрична, ручна, живлення від мережі",
     "jargon": True,
     "note": "жаргон: тариф знає лише «інструменти ручні електромеханічні»"},
    {"q": "паровий димогарний котел",
     "expected": "8402191000", "followup": None, "jargon": False,
     "note": "контроль: пряме влучання без переформулювань"},
    {"q": "жіноча сумка з натуральної шкіри",
     "expected": "4202210000", "followup": None, "jargon": False,
     "note": "перевірка CANDIDATES_FOR_LLM=15: раніше еталон обрізався на 10"},
]


class Recorder:
    """Обгортає інструменти агента: пише всі пошукові запити і фінальний код."""
    def __init__(self, agent):
        self.searches, self.finalized = [], None
        self._s, self._f = agent._tool_search, agent._tool_finalize
        agent._tool_search = self.search
        agent._tool_finalize = self.finalize
    def search(self, query):
        self.searches.append(query)
        return self._s(query)
    def finalize(self, code, reasoning):
        out = self._f(code, reasoning)
        if out.get("ok"):
            self.finalized = out["code"]
        return out


def run_case(retriever, client, case, max_followups=1):
    agent = UktzedAgent(retriever, client)
    rec = Recorder(agent)
    reply = agent.turn(case["q"])
    followups = 0
    while rec.finalized is None and case["followup"] and followups < max_followups:
        followups += 1
        reply = agent.turn(case["followup"])
    ok_code = rec.finalized is not None and rec.finalized.startswith(case["expected"])
    ok_jargon = (not case["jargon"]) or len(rec.searches) >= 2
    return {
        "query": case["q"], "expected": case["expected"], "final": rec.finalized,
        "searches": rec.searches, "followups_used": followups,
        "pass": ok_code and ok_jargon,
        "fail_reason": (None if ok_code and ok_jargon else
                        ("код не збігся" if not ok_code else
                         "жаргон: не було повторного пошуку (правило 2a не спрацювало)")),
        "reply_tail": (reply or "")[-300:],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", action="store_true")
    ap.add_argument("--plumbing", action="store_true", help="перевірка харнесу без API")
    ap.add_argument("--jsonl", default="uktzed_nodes.jsonl")
    a = ap.parse_args()

    r = UktzedRetriever(a.jsonl, use_dense=a.dense)

    if a.plumbing:
        agent = UktzedAgent(r, None)
        rec = Recorder(agent)
        s = agent._tool_search("димогарний котел")
        f = agent._tool_finalize("8402191000", "тест")
        assert rec.searches == ["димогарний котел"] and rec.finalized == "8402191000", (rec.searches, rec.finalized)
        assert agent._tool_finalize("9999999999", "тест")["ok"] is False
        print("plumbing OK: рекордер ловить пошуки і фінальний код, верифікація жива.")
        return

    client = make_client()
    print(f"Модель: {MODEL} · режим: {'e5' if a.dense else 'лексика'} · кейсів: {len(CASES)}\n")
    results = [run_case(r, client, c) for c in CASES]

    fails = 0
    for res in results:
        mark = "PASS" if res["pass"] else "FAIL"
        fails += (not res["pass"])
        print(f"[{mark}] «{res['query']}»")
        print(f"       очікувано {res['expected']}* → отримано {res['final']}")
        print(f"       пошуки ({len(res['searches'])}): {res['searches']}")
        if res["followups_used"]:
            print(f"       уточнень використано: {res['followups_used']}")
        if res["fail_reason"]:
            print(f"       причина: {res['fail_reason']}")
            print(f"       хвіст відповіді: …{res['reply_tail']}")
        print()
    print(f"Підсумок: {len(results)-fails}/{len(results)} PASS")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
