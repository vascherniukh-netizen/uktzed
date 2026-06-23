# -*- coding: utf-8 -*-
"""
Парсер Митного тарифу України (УКТ ЗЕД) з HTML-вивантажень rada.
Будує ієрархічне дерево й денормалізовані листя з повним шляхом
(Розділ / Глава / підрозділ / позиція / ... / лист) для RAG.

Залежності: beautifulsoup4 (чистий Python, працює на Windows cmd без компілятора).
Парсер: вбудований 'html.parser' зі стандартної бібліотеки — без lxml.
"""
import re, json, sys, glob, os
from bs4 import BeautifulSoup

DASH_RE   = re.compile(r'^\s*((?:[-–]\s*)+)')          # провідні тире (звичайне й тире-en)
ROMAN_RE  = re.compile(r'^\s*([IVXLCDM]+)\.\s+\S')      # римський підрозділ всередині глави
ROZDIL_RE = re.compile(r'^\s*Розділ\s+[IVXLCDM]+', re.I)
GRUPA_RE  = re.compile(r'^\s*(?:Група|Глава)\s+\d', re.I)

def clean(s):
    return re.sub(r'\s+', ' ', (s or '')).replace('\xa0', ' ').strip()

def dash_depth(name):
    m = DASH_RE.match(name)
    return m.group(1).count('-') + m.group(1).count('–') if m else 0

def norm_code(raw):
    c = re.sub(r'\s+', '', raw or '')
    return c if c.isdigit() else None

def is_tariff_table(tbl):
    head = tbl.find('tr')
    if not head: return False
    t = clean(head.get_text(' '))
    return 'Код' in t and 'Назва' in t

def rate_layout(tbl):
    # к-сть колонок ставок визначаємо з підзаголовка таблиці
    for tr in tbl.find_all('tr')[:3]:
        cells = [clean(c.get_text(' ')) for c in tr.find_all(['td', 'th'])]
        joined = ' '.join(cells).lower()
        if 'пільгова' in joined or 'повна' in joined or 'преференц' in joined:
            n = sum(k in joined for k in ('преференц', 'пільгова', 'повна'))
            return max(n, 2)
    return 2

def heading_candidates(soup):
    # елементи-заголовки поза таблицями, у документному порядку
    out = []
    for el in soup.find_all(['p', 'div', 'h1', 'h2', 'h3', 'h4', 'table']):
        if el.name == 'table':
            out.append(('table', el)); continue
        if el.find_parent('table') is not None:
            continue
        txt = clean(el.get_text(' '))
        if ROZDIL_RE.match(txt) and len(txt) < 200:
            out.append(('rozdil', txt))
        elif GRUPA_RE.match(txt) and len(txt) < 200:
            out.append(('grupa', txt))
    return out

def parse_file(path, nodes):
    soup = BeautifulSoup(open(path, encoding='utf-8').read(), 'html.parser')
    rozdil = grupa = subchapter = None
    for kind, payload in heading_candidates(soup):
        if kind == 'rozdil':
            rozdil, grupa, subchapter = payload, None, None; continue
        if kind == 'grupa':
            grupa, subchapter = payload, None; continue
        tbl = payload
        if not is_tariff_table(tbl):
            continue
        nrate = rate_layout(tbl)
        stack = []  # stack[d] = index у nodes для dash-глибини d
        for tr in tbl.find_all('tr'):
            cells = [clean(c.get_text(' ')) for c in tr.find_all('td')]
            if len(cells) < 2:
                continue
            code_raw, name = cells[0], cells[1]
            if not name or name.lower().startswith(('код', 'пільгова', 'повна', 'преференц')):
                continue
            code = norm_code(code_raw)
            rates = cells[2:2 + nrate]
            ovo = cells[2 + nrate] if len(cells) > 2 + nrate else ''

            # римський підрозділ всередині глави (порожній код)
            if code is None and ROMAN_RE.match(name):
                subchapter = name; stack = []; continue

            d = dash_depth(name)
            del stack[d:]                       # обрізаємо до глибини
            parent = stack[d - 1] if d > 0 and d - 1 < len(stack) else None
            ctx = [x for x in (rozdil, grupa, subchapter) if x]
            anc_names = [nodes[i]['name'] for i in stack[:d]]
            node = {
                'code': code,
                'name': name,
                'depth': d,
                'rozdil': rozdil, 'grupa': grupa, 'subchapter': subchapter,
                'rates': {k: v for k, v in zip(('преференційна', 'пільгова', 'повна')[-nrate:], rates) if v},
                'ovo': ovo if ovo and ovo != '-' else None,
                'path': ctx + anc_names + [name],
                'parent': parent,
                'n_children': 0,
            }
            idx = len(nodes); nodes.append(node)
            if parent is not None:
                nodes[parent]['n_children'] += 1
            if d < len(stack):
                stack[d] = idx
            else:
                stack.append(idx)
    return nodes

def main():
    files = sys.argv[1:] or glob.glob('*.html') or glob.glob('/mnt/user-data/uploads/*.html')
    if not files:
        print('Не знайдено .html у поточній теці. Поклади два файли тарифу поруч зі скриптом.')
        return
    nodes = []
    for f in files:
        parse_file(f, nodes)
    # лист = має код і не має дочірніх вузлів
    for n in nodes:
        n['is_leaf'] = bool(n['code']) and n['n_children'] == 0
        # денормалізований текст для ембедингу
        n['chunk'] = ' / '.join(n['path'])
    coded  = [n for n in nodes if n['code']]
    leaves = [n for n in nodes if n['is_leaf']]
    out = 'uktzed_nodes.jsonl'
    with open(out, 'w', encoding='utf-8') as fh:
        for n in nodes:
            fh.write(json.dumps(n, ensure_ascii=False) + '\n')

    print('файлів оброблено      :', len(files))
    print('усього вузлів         :', len(nodes))
    print('з кодом               :', len(coded))
    print('листів (фінальні коди):', len(leaves))
    print('макс. глибина тире    :', max(n['depth'] for n in nodes))
    lens = {}
    for n in coded:
        L = len(n['code']); lens[L] = lens.get(L, 0) + 1
    print('довжини кодів         :', dict(sorted(lens.items())))
    print('\n--- приклади листів (повний денормалізований шлях) ---')
    import random; random.seed(1)
    for n in random.sample(leaves, min(3, len(leaves))):
        print(f"\nКод {n['code']}  ставки={n['rates']}  ОВО={n['ovo']}")
        print('  chunk:', n['chunk'])

if __name__ == '__main__':
    main()
