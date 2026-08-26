# -*- coding: utf-8 -*-
"""
Генерація побутових/торгових синонімів для 4-значних позицій УКТ ЗЕД через DeepSeek.

Результат: uktzed_synonyms.json {"8523": ["флешка", ...], ...}
Ретривер підхоплює його автоматично і збагачує ЛИШЕ лексичні індекси
(e5-кеш лишається валідним — dense-тексти не змінюються).

~1250 позицій · ~1 виклик на позицію · орієнтовно <$1 на deepseek-v4-flash.
Скрипт відновлюваний: уже згенеровані позиції пропускаються, файл пишеться
кожні 25 позицій. Обрив у будь-який момент безпечний.

Запуск:
    python uktzed_enrich.py --dry-run   # показати 3 промти, без API
    python uktzed_enrich.py             # повна генерація (ключ з .env)
"""
import os, sys, json, argparse, threading, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from uktzed_agent import _load_dotenv, BASE_URL   # тягне і .env-лоадер

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

MODEL = os.environ.get('UKTZED_MODEL', 'deepseek-v4-flash')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uktzed_synonyms.json')

PROMPT = """Товарна позиція {pos} Митного тарифу України:
«{header}»

Приклади підпозицій: {children}

Напиши 6-14 назв, якими ЗВИЧАЙНІ ЛЮДИ та ІМПОРТЕРИ називають товари цієї позиції:
побутові слова, торгові назви категорій, професійний жаргон, поширені англіцизми
(латиницею, якщо так кажуть). НЕ повторюй слова з офіційного тексту вище.
СТРОГО УКРАЇНСЬКОЮ: жодного російського слова, жодного суржику чи транслітерованого
сміття (типу «вєщі», «лєдіс», «шмотки»). Англіцизми — лише латиницею.
Тільки список через кому, без пояснень і нумерації. Якщо позиція суто технічна
і побутових назв не має — напиши лише найужитковіші 2-3 родові слова."""


def build_tasks(jsonl):
    nodes = [json.loads(l) for l in open(jsonl, encoding='utf-8')]
    leaves = [n for n in nodes if n.get('is_leaf') and n.get('code')]
    by4 = {}
    for n in leaves:
        by4.setdefault(n['code'][:4], []).append(n)
    tasks = []
    for pos, ns in sorted(by4.items()):
        header = ns[0]['path'][2] if len(ns[0]['path']) > 2 else ns[0]['name']
        kids = []
        for n in ns:
            nm = n['name'].strip('- ').strip()
            if nm and nm.lower() not in ('інше', 'інші') and nm not in kids:
                kids.append(nm)
            if len(kids) >= 8:
                break
        tasks.append((pos, header[:600], '; '.join(kids)[:500] or '—'))
    return tasks


RU_RE = re.compile(r'[ыэъё]', re.I)

def parse_terms(text):
    parts = re.split(r'[,\n;]+', text or '')
    out = []
    for p in parts:
        t = p.strip(' .«»"\'-–—').strip()
        if RU_RE.search(t):                       # російські маркери — геть
            continue
        if 1 < len(t) <= 60 and not t.lower().startswith(('приклад', 'назв')):
            out.append(t.lower())
    seen = set(); res = []
    for t in out:
        if t not in seen:
            seen.add(t); res.append(t)
    return res[:14]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uktzed_nodes.jsonl'))
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--threads', type=int, default=6)
    a = ap.parse_args()

    tasks = build_tasks(a.jsonl)
    done = {}
    if os.path.exists(OUT):
        done = json.load(open(OUT, encoding='utf-8'))
        print(f"продовжую: {len(done)} позицій уже є")
    done = {k: v for k, v in done.items() if v}      # порожні списки = не зроблено
    todo = [t for t in tasks if t[0] not in done]
    print(f"позицій всього: {len(tasks)}, до генерації: {len(todo)}")

    if a.dry_run:
        for pos, header, kids in todo[:3]:
            print('='*80)
            print(PROMPT.format(pos=pos, header=header, children=kids))
        return

    from openai import OpenAI
    key = os.environ.get('DEEPSEEK_API_KEY') or os.environ.get('UKTZED_API_KEY')
    if not key:
        sys.exit('Не задано DEEPSEEK_API_KEY (у .env або середовищі).')
    client = OpenAI(api_key=key, base_url=BASE_URL)
    lock = threading.Lock()
    usage = {'in': 0, 'out': 0}

    def one(task):
        pos, header, kids = task
        resp = client.chat.completions.create(
            model=MODEL, max_tokens=300, temperature=1.0,
            # v4-flash — reasoning-модель: без цього весь бюджет з'їдає міркування,
            # а content лишається порожнім (finish_reason=length). Перевірено 2026-08-26.
            extra_body={"reasoning_effort": "none"},
            messages=[{'role': 'user',
                       'content': PROMPT.format(pos=pos, header=header, children=kids)}])
        u = resp.usage
        with lock:
            usage['in'] += u.prompt_tokens; usage['out'] += u.completion_tokens
        terms = parse_terms(resp.choices[0].message.content)
        if not terms:
            raise RuntimeError(f"порожня відповідь для {pos} "
                               f"(finish_reason={resp.choices[0].finish_reason})")
        return pos, terms

    n_done = 0
    with ThreadPoolExecutor(max_workers=a.threads) as ex:
        futs = [ex.submit(one, t) for t in todo]
        for f in as_completed(futs):
            try:
                pos, terms = f.result()
            except Exception as e:
                print(f"  ! помилка ({e.__class__.__name__}: {e}) — позицію пропущено, перезапустиш пізніше")
                continue
            done[pos] = terms
            n_done += 1
            if n_done % 25 == 0:
                with lock:
                    json.dump(done, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
                print(f"  {n_done}/{len(todo)} · токени in={usage['in']} out={usage['out']}")
    json.dump(done, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    cost = usage['in']/1e6*0.28 + usage['out']/1e6*0.42   # орієнтовні ціни v4-flash
    print(f"Готово: {len(done)} позицій у {os.path.basename(OUT)}. "
          f"Токени: in={usage['in']}, out={usage['out']} (~${cost:.2f})")

if __name__ == '__main__':
    main()
