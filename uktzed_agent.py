# -*- coding: utf-8 -*-
"""
ОПІ-агент УКТ ЗЕД: Claude як conversational-шар над детермінованим ретривером.

Потік:
  опис товару → search_uktzed (ретривер дає кандидатів з повними шляхами)
              → Claude застосовує ОПІ; якщо неоднозначно — ставить ОДНЕ уточнююче
                питання; інакше → finalize_code
              → finalize_code верифікує, що код є в наборі (анти-галюцинація),
                і повертає офіційний запис (ставки, ОВО).

Залежності: anthropic (+ ретривер тягне sklearn/numpy).
    pip install anthropic
Ключ: змінна середовища ANTHROPIC_API_KEY (у код не вшивається).
    Windows cmd:  set ANTHROPIC_API_KEY=sk-ant-...

Запуск:
    python uktzed_agent.py            # діалог
    python uktzed_agent.py --selftest # перевірка плумбінгу без API-ключа
"""
import os, sys, json, argparse
from uktzed_retriever import UktzedRetriever

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

MODEL = os.environ.get('UKTZED_MODEL', 'claude-sonnet-4-6')

SYSTEM = """Ти — асистент з класифікації товарів за УКТ ЗЕД (Митний тариф України, Закон 2697-IX).
Твоє завдання — визначити правильний 10-значний код для опису товару від користувача.

Жорсткі правила:
1. Коди НЕ вигадуй. Працюй ВИКЛЮЧНО з кандидатами, які повернув інструмент search_uktzed.
   Фінальний код можна вказати лише через finalize_code, і лише якщо він був серед результатів пошуку.
2. Спочатку завжди викликай search_uktzed з коротким описом товару (іменники-ознаки, без стоп-слів).
3. Дивись на ПОВНИЙ шлях кожного кандидата (Розділ / Група / позиція / ... / лист), а не лише на назву листа.
   Застосовуй Основні правила інтерпретації (ОПІ): класифікуй за суттю виробу, матеріалом, ступенем обробки,
   призначенням та комплектністю.
4. Якщо кандидати розходяться по одній вирішальній ознаці (напр. матеріал верху взуття; готовий виріб чи тканина;
   для роздрібу чи ні; новий чи вживаний) — постав КОРИСТУВАЧЕВІ рівно ОДНЕ коротке уточнююче питання саме по цій
   ознаці. Не став більше одного питання за раз. Після відповіді — за потреби зроби новий, уточнений search_uktzed.
5. Коли впевнений — виклич finalize_code з обраним кодом і одним реченням обґрунтування (яка ознака вирішила).
6. У фінальній відповіді стисло наведи: код, найменування, ставки мита, і одне речення ОПІ-обґрунтування.
   Додай короткий дисклеймер, що остаточну класифікацію підтверджує митний брокер/митниця (ст. 69 МКУ).

Відповідай українською, без зайвої води."""

TOOLS = [
    {
        "name": "search_uktzed",
        "description": "Пошук кандидатів коду УКТ ЗЕД за описом товару живою мовою. "
                       "Повертає до 8 листів із повним ієрархічним шляхом, ставками та ОВО.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "Опис товару українською (ознаки: тип, матеріал, обробка, призначення)."}
            },
            "required": ["query"],
        },
    },
    {
        "name": "finalize_code",
        "description": "Зафіксувати остаточний код. Приймається ЛИШЕ код, що існує в наборі УКТ ЗЕД. "
                       "Повертає офіційний запис (найменування, ставки, ОВО) або помилку, якщо коду немає.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "10-значний код без пробілів."},
                "reasoning": {"type": "string", "description": "Одне речення: яка ознака вирішила вибір."},
            },
            "required": ["code", "reasoning"],
        },
    },
]


class UktzedAgent:
    CANDIDATES_FOR_LLM = 10           # скільки кандидатів бачить Claude (менше = менше токенів)

    def __init__(self, retriever, client):
        self.r = retriever
        self.client = client
        self.messages = []

    @staticmethod
    def _short_path(chunk):
        # Стислий контекст замість повного ланцюга від кореня:
        # назва глави + два останні рівні (товарна позиція → лист).
        parts = [p.strip() for p in chunk.split(' / ') if p.strip()]
        grupa = next((p for p in parts if p.lower().startswith(('група', 'глава'))), None)
        tail = parts[-2:]
        out = ([grupa] if grupa else []) + [p for p in tail if p != grupa]
        return ' / '.join(out)

    # --- детерміновані інструменти ---
    def _tool_search(self, query):
        hits = self.r.search(query, top_k=20)            # шукаємо глибше для повноти
        hits = hits[:self.CANDIDATES_FOR_LLM]            # але Claude бачить лише верхівку
        if not hits:
            return {"candidates": [], "note": "Збігів не знайдено — уточни опис товару."}
        # коди + стислий шлях; ставки сюди НЕ кладемо (вони потрібні лише для фінального коду)
        return {"candidates": [{"code": h["code"], "path": self._short_path(h["path"])}
                               for h in hits]}

    def _tool_finalize(self, code, reasoning):
        code = (code or "").replace(" ", "")
        rec = self.r.canon.get(code)               # ВЕРИФІКАЦІЯ: код мусить існувати
        if rec is None:
            return {"ok": False,
                    "error": f"Код {code} відсутній у наборі УКТ ЗЕД. "
                             f"Обери код ЛИШЕ зі списку search_uktzed."}
        return {"ok": True, "code": code, "name": rec["name"],
                "path": rec["chunk"], "rates": rec.get("rates"),
                "ovo": rec.get("ovo"), "reasoning": reasoning}

    def _dispatch(self, name, inp):
        if name == "search_uktzed":
            return self._tool_search(inp.get("query", ""))
        if name == "finalize_code":
            return self._tool_finalize(inp.get("code", ""), inp.get("reasoning", ""))
        return {"error": f"невідомий інструмент {name}"}

    # --- один хід діалогу: повертає фінальний текст асистента ---
    def turn(self, user_text):
        self.messages.append({"role": "user", "content": user_text})
        while True:
            resp = self.client.messages.create(
                model=MODEL, max_tokens=1024, system=SYSTEM,
                tools=TOOLS, messages=self.messages,
            )
            self.messages.append({"role": "assistant", "content": resp.content})
            tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
            if not tool_uses:                       # модель відповіла текстом → хід завершено
                return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
            results = []
            for tu in tool_uses:
                out = self._dispatch(tu.name, tu.input)
                results.append({"type": "tool_result", "tool_use_id": tu.id,
                                "content": json.dumps(out, ensure_ascii=False)})
            self.messages.append({"role": "user", "content": results})


def make_client():
    from anthropic import Anthropic
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        sys.exit("Не задано ANTHROPIC_API_KEY. У cmd:  set ANTHROPIC_API_KEY=sk-ant-...")
    return Anthropic(api_key=key)


def chat(jsonl, use_dense=False):
    r = UktzedRetriever(jsonl, use_dense=use_dense)
    print(f"Індекс: {len(r.leaves)} листів. Опиши товар (порожній рядок — вихід).\n")
    agent = UktzedAgent(r, make_client())
    while True:
        try:
            q = input("товар> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        print("\n" + agent.turn(q) + "\n")


# ---------- selftest: плумбінг без API ----------
class _FakeBlock:
    def __init__(self, **k): self.__dict__.update(k)

class _FakeResp:
    def __init__(self, content): self.content = content

class _FakeClient:
    """Імітує дві відповіді моделі: search → finalize → текст. Перевіряє диспетч і верифікацію."""
    def __init__(self): self.step = 0
    class messages:
        pass
    def __init__(self):
        self.step = 0
        self.messages = self
    def create(self, **kw):
        self.step += 1
        if self.step == 1:
            return _FakeResp([_FakeBlock(type="tool_use", id="t1", name="search_uktzed",
                                         input={"query": "димогарний котел"})])
        if self.step == 2:
            return _FakeResp([_FakeBlock(type="tool_use", id="t2", name="finalize_code",
                                         input={"code": "8402191000", "reasoning": "котли димогарні"})])
        return _FakeResp([_FakeBlock(type="text",
                                     text="Код 8402191000 — котли димогарні. Ставки 5%/5%. "
                                          "Підтверджує брокер/митниця.")])

def selftest(jsonl):
    r = UktzedRetriever(jsonl)
    a = UktzedAgent(r, _FakeClient())
    # 1) search повертає реальних кандидатів
    s = a._tool_search("димогарний котел")
    assert s["candidates"] and s["candidates"][0]["code"] == "8402191000", s
    # 2) finalize реального коду — ок
    f = a._tool_finalize("8402 19 10 00", "тест")
    assert f["ok"] and f["code"] == "8402191000", f
    # 3) finalize вигаданого коду — відхилено (анти-галюцинація)
    bad = a._tool_finalize("9999999999", "тест")
    assert bad["ok"] is False, bad
    # 4) повний цикл через фейковий клієнт
    out = a.turn("димогарний котел")
    assert "8402191000" in out, out
    print("selftest OK: пошук, верифікація, відхилення фейку, повний цикл — усе працює.")
    print("  приклад фінального тексту:", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dense", action="store_true", help="увімкнути e5 (потрібна модель)")
    ap.add_argument("--jsonl", default=os.path.join(os.path.dirname(__file__) or ".", "uktzed_nodes.jsonl"))
    a = ap.parse_args()
    if a.selftest:
        selftest(a.jsonl)
    else:
        chat(a.jsonl, use_dense=a.dense)


if __name__ == "__main__":
    main()
