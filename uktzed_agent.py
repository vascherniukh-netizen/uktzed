# -*- coding: utf-8 -*-
"""
ОПІ-агент УКТ ЗЕД: LLM як conversational-шар над детермінованим ретривером.

LLM-провайдер: DeepSeek (нативний OpenAI-сумісний формат, tool calls).
Ретривер (BM25 + char-TF-IDF + локальні e5-ембединги) — без змін, працює офлайн.

Потік:
  опис товару → search_uktzed (ретривер дає кандидатів з повними шляхами)
              → модель застосовує ОПІ; якщо неоднозначно — ставить ОДНЕ уточнююче
                питання; інакше → finalize_code
              → finalize_code верифікує, що код є в наборі (анти-галюцинація),
                і повертає офіційний запис (ставки, ОВО).

Залежності: openai (+ ретривер тягне sklearn/numpy).
    pip install openai
Ключ: змінна середовища DEEPSEEK_API_KEY (у код не вшивається).
    Windows cmd:  set DEEPSEEK_API_KEY=sk-...

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

def _load_dotenv():
    """Мінімальний .env-лоадер (без залежностей). Ключ НЕ вшивається в код і не
    потрапляє в git — .env уже в .gitignore. Змінні з середовища мають пріоритет."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if not os.path.exists(path):
        return
    for line in open(path, encoding='utf-8-sig'):
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


_load_dotenv()

MODEL    = os.environ.get('UKTZED_MODEL', 'deepseek-v4-flash')
BASE_URL = os.environ.get('UKTZED_BASE_URL', 'https://api.deepseek.com')
MAX_ROUNDS = 8          # запобіжник від нескінченного циклу tool-викликів

SYSTEM = """Ти — асистент з класифікації товарів за УКТ ЗЕД (Митний тариф України, Закон 2697-IX).
Твоє завдання — визначити правильний 10-значний код для опису товару від користувача.

Жорсткі правила:
1. Коди НЕ вигадуй. Працюй ВИКЛЮЧНО з кандидатами, які повернув інструмент search_uktzed.
   Фінальний код можна вказати лише через finalize_code, і лише якщо він був серед результатів пошуку.
2. Спочатку завжди викликай search_uktzed з коротким описом товару (іменники-ознаки, без стоп-слів).
2a. МОВА ТАРИФУ, НЕ ЖАРГОН. Тариф не знає торгових назв, сленгу і професійного жаргону.
   Якщо кандидати виглядають не по темі (інша галузь, інший тип виробу) — не здавайся і не бери
   найкраще з поганого. Переформулюй запит родовими термінами тарифу і виклич search_uktzed ЩЕ РАЗ.
   Приклади: «маяк-пінгер» → «прилад навігаційний»; «болгарка» → «машина шліфувальна ручна
   електромеханічна»; «павербанк» → «акумулятор літій-іонний». Роби до 3 спроб пошуку за хід.
3. Дивись на ПОВНИЙ шлях кожного кандидата (Розділ / Група / позиція / ... / лист), а не лише на назву листа.
   Застосовуй Основні правила інтерпретації (ОПІ): класифікуй за суттю виробу, матеріалом, ступенем обробки,
   призначенням та комплектністю.
4. Якщо кандидати розходяться по одній вирішальній ознаці (напр. матеріал верху взуття; готовий виріб чи тканина;
   для роздрібу чи ні; новий чи вживаний) — постав КОРИСТУВАЧЕВІ рівно ОДНЕ коротке уточнююче питання саме по цій
   ознаці. Не став більше одного питання за раз. Після відповіді — за потреби зроби новий, уточнений search_uktzed.
5. Коли впевнений — виклич finalize_code з обраним кодом і одним реченням обґрунтування (яка ознака вирішила).
6. ФІНАЛЬНА ВІДПОВІДЬ — строго в такому порядку:
   • Код УКТ ЗЕД
   • «Офіційне найменування (дослівно з тарифу)» — рівні з поля official_chain, ДОСЛІВНО, через « / ».
     Це текст закону: НЕ перефразовуй, НЕ скорочуй, НЕ дописуй пояснень у дужках, НЕ виправляй
     тире й двокрапки. Якщо рівень довший за ~200 символів — обірви його і постав «…»
     (чесний маркер скорочення), але не переписуй своїми словами.
   • «Простими словами» — один рядок власного переказу. Явно позначений як переказ, не як назва з тарифу.
   • Ставки мита і ОВО — з полів rates та ovo, як є.
   • Одне речення ОПІ-обґрунтування: яка ознака вирішила.
   • Дисклеймер: остаточну класифікацію підтверджує митний брокер/митниця (ст. 69 МКУ).
7. Ставки, ОВО й офіційне найменування бери ВИКЛЮЧНО з відповіді finalize_code. Своїх чисел не додавай.

Відповідай українською, без зайвої води."""

# OpenAI-формат tool schemas (DeepSeek native)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_uktzed",
            "description": "Пошук кандидатів коду УКТ ЗЕД за описом товару живою мовою. "
                           "Повертає до 10 листів із повним ієрархічним шляхом.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string",
                              "description": "Опис товару українською (ознаки: тип, матеріал, обробка, призначення)."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize_code",
            "description": "Зафіксувати остаточний код. Приймається ЛИШЕ код, що існує в наборі УКТ ЗЕД. "
                           "Повертає офіційний запис (найменування, ставки, ОВО) або помилку, якщо коду немає.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "10-значний код без пробілів."},
                    "reasoning": {"type": "string", "description": "Одне речення: яка ознака вирішила вибір."},
                },
                "required": ["code", "reasoning"],
            },
        },
    },
]


class UktzedAgent:
    # Скільки кандидатів бачить модель. Ретривер шукає до top_k=20, але показувати всі — дорого.
    # Заміряно на uktzed_eval_seed.jsonl з e5: еталон у топ-10 → 86.7% кейсів, у топ-15 → 93.3%.
    # Обрізка на 10 викидала, напр., «жіночу сумку» (еталон стояв 11-м).
    CANDIDATES_FOR_LLM = int(os.environ.get('UKTZED_CANDIDATES', 15))

    def __init__(self, retriever, client):
        self.r = retriever
        self.client = client
        self.messages = [{"role": "system", "content": SYSTEM}]

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
        query = (query or "").strip()
        if not query:                                    # модель інколи шле порожній виклик
            return {"candidates": [], "note": "Порожній запит — передай опис товару словами."}
        hits = self.r.search(query, top_k=20)            # шукаємо глибше для повноти
        hits = hits[:self.CANDIDATES_FOR_LLM]            # але модель бачить лише верхівку
        if not hits:
            return {"candidates": [], "note": "Збігів не знайдено — уточни опис товару."}
        return {"candidates": [{"code": h["code"], "path": self._short_path(h["path"])}
                               for h in hits]}

    def _tool_finalize(self, code, reasoning):
        code = (code or "").replace(" ", "")
        rec = self.r.canon.get(code)               # ВЕРИФІКАЦІЯ: код мусить існувати
        if rec is None:
            return {"ok": False,
                    "error": f"Код {code} відсутній у наборі УКТ ЗЕД. "
                             f"Обери код ЛИШЕ зі списку search_uktzed."}
        path = rec.get("path") or []
        # офіційний ланцюжок: рівні НИЖЧЕ Розділу і Групи (товарна позиція → ... → лист),
        # дослівно як у тарифі — саме це має цитуватись у декларації
        chain = [p for p in path[2:]] if len(path) > 2 else [rec["name"]]
        return {"ok": True, "code": code,
                "name": rec["name"],                    # дослівна назва листа
                "official_chain": chain,                # дослівні рівні тарифу
                "official_name": " / ".join(chain),
                "rozdil": rec.get("rozdil"), "grupa": rec.get("grupa"),
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
        for _ in range(MAX_ROUNDS):
            resp = self.client.chat.completions.create(
                model=MODEL, max_tokens=1024, messages=self.messages,
                tools=TOOLS, tool_choice="auto",
            )
            msg = resp.choices[0].message
            calls = list(getattr(msg, "tool_calls", None) or [])

            # відправляємо назад лише те, що приймає API (без reasoning_content тощо)
            am = {"role": "assistant", "content": msg.content or ""}
            if calls:
                am["tool_calls"] = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in calls
                ]
            self.messages.append(am)

            if not calls:                       # модель відповіла текстом → хід завершено
                return msg.content or ""

            for tc in calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                out = self._dispatch(tc.function.name, args)
                self.messages.append({"role": "tool", "tool_call_id": tc.id,
                                      "content": json.dumps(out, ensure_ascii=False)})
        return "Не вдалося завершити класифікацію за відведену кількість кроків — уточни опис товару."


def make_client():
    from openai import OpenAI
    key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("UKTZED_API_KEY")
    if not key:
        sys.exit("Не задано DEEPSEEK_API_KEY. У cmd:  set DEEPSEEK_API_KEY=sk-...")
    return OpenAI(api_key=key, base_url=BASE_URL)


def chat(jsonl, use_dense=False):
    r = UktzedRetriever(jsonl, use_dense=use_dense)
    print(f"Індекс: {len(r.leaves)} листів · модель: {MODEL}. Опиши товар (порожній рядок — вихід).\n")
    client = make_client()
    agent = UktzedAgent(r, client)
    while True:
        try:
            q = input("товар> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        print("\n" + agent.turn(q) + "\n")


# ---------- selftest: плумбінг без API ----------
class _FakeFn:
    def __init__(self, name, arguments): self.name, self.arguments = name, arguments

class _FakeCall:
    def __init__(self, id, name, arguments): self.id, self.function = id, _FakeFn(name, arguments)

class _FakeMsg:
    def __init__(self, content=None, tool_calls=None):
        self.content, self.tool_calls, self.role = content, tool_calls, "assistant"

class _FakeChoice:
    def __init__(self, message): self.message = message

class _FakeResp:
    def __init__(self, message): self.choices = [_FakeChoice(message)]

class _FakeCompletions:
    def __init__(self, outer): self.outer = outer
    def create(self, **kw):
        self.outer.step += 1
        self.outer.last_kw = kw
        if self.outer.step == 1:
            return _FakeResp(_FakeMsg(tool_calls=[_FakeCall(
                "t1", "search_uktzed", json.dumps({"query": "димогарний котел"}, ensure_ascii=False))]))
        if self.outer.step == 2:
            return _FakeResp(_FakeMsg(tool_calls=[_FakeCall(
                "t2", "finalize_code",
                json.dumps({"code": "8402191000", "reasoning": "котли димогарні"}, ensure_ascii=False))]))
        return _FakeResp(_FakeMsg(content="Код 8402191000 — котли димогарні. Ставки 5%/5%. "
                                          "Підтверджує брокер/митниця."))

class _FakeClient:
    """Імітує OpenAI-клієнт: search → finalize → текст. Перевіряє диспетч і верифікацію."""
    def __init__(self):
        self.step = 0
        self.last_kw = None
        self.chat = self
        self.completions = _FakeCompletions(self)


def selftest(jsonl):
    r = UktzedRetriever(jsonl)
    a = UktzedAgent(r, _FakeClient())
    # 1) search повертає реальних кандидатів
    empty = a._tool_search("   ")
    assert empty["candidates"] == [] and "Порожній" in empty["note"], empty
    s = a._tool_search("димогарний котел")
    assert s["candidates"] and s["candidates"][0]["code"] == "8402191000", s
    # 2) finalize реального коду — ок
    f = a._tool_finalize("8402 19 10 00", "тест")
    assert f["ok"] and f["code"] == "8402191000", f
    # 2b) офіційний ланцюжок — дослівні рівні тарифу, без парафразу
    f2 = a._tool_finalize("6403190000", "тест")
    assert f2["official_chain"][-1] == "- - інше", f2["official_chain"]
    assert f2["official_chain"][0].startswith("Взуття на підошві з гуми"), f2["official_chain"]
    assert not f2["official_chain"][0].startswith("Розділ"), "Розділ/Група не мають бути в ланцюжку"
    assert f2["name"] == "- - інше", f2["name"]
    # 3) finalize вигаданого коду — відхилено (анти-галюцинація)
    bad = a._tool_finalize("9999999999", "тест")
    assert bad["ok"] is False, bad
    # 4) повний цикл через фейковий клієнт
    out = a.turn("димогарний котел")
    assert "8402191000" in out, out
    # 5) формат історії відповідає OpenAI/DeepSeek: system → user → assistant(tool_calls) → tool → …
    roles = [m["role"] for m in a.messages]
    assert roles[:4] == ["system", "user", "assistant", "tool"], roles
    assert a.messages[2]["tool_calls"][0]["type"] == "function", a.messages[2]
    assert a.messages[3]["tool_call_id"] == "t1", a.messages[3]
    # 6) схеми інструментів у OpenAI-формі
    assert all(t["type"] == "function" and "parameters" in t["function"] for t in TOOLS)
    print("selftest OK: пошук, верифікація, відхилення фейку, повний цикл, OpenAI-формат — усе працює.")
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
