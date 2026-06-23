# -*- coding: utf-8 -*-
"""
Гібридний ретривер коду УКТ ЗЕД по денормалізованих листях (uktzed_nodes.jsonl).

Лексика  : BM25 по словах (власна реалізація, без сторонніх залежностей).
Морфологія: char n-gram TF-IDF (3-5), ловить словозміну й одруки без жодної моделі.
Злиття   : Reciprocal Rank Fusion (RRF).
Верифікація: кожен повернутий код звіряється з канонічним набором кодів.

Залежність одна: scikit-learn (Windows wheels, без компіляторів). BM25 — власний.
Продакшн-апгрейд: клас DenseE5Index нижче — багатомовні ембединги intfloat/multilingual-e5,
вмикаються локально (потрібне завантаження моделі), API той самий.

Запуск (Windows cmd):
    python uktzed_retriever.py                 # демо-запити
    python uktzed_retriever.py "жіноча вовняна сукня"
    python uktzed_retriever.py -i              # інтерактивно
"""
import sys, os, re, json, math, argparse, hashlib
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

TOKEN_RE = re.compile(r'[\w’\'-]+', re.UNICODE)
STOP = {'та','і','й','з','із','зі','для','або','чи','у','в','на','до','що','крім',
        'інше','інші','інша','інший','розділ','група','глава'}
LABEL_RE = re.compile(r'^(Розділ\s+[IVXLCDM]+|Група\s+\d+|Глава\s+\d+)\s+', re.I)

def tokenize(text):
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOP and len(t) > 1]

def clean_chunk(chunk):
    # прибираємо службові префікси "Розділ XI " / "Група 62 ", лишаючи назви
    parts = [LABEL_RE.sub('', p).strip() for p in chunk.split(' / ')]
    return ' / '.join(p for p in parts if p)

# ---------- BM25 (власна реалізація) ----------
class BM25:
    def __init__(self, docs_tokens, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.N = len(docs_tokens)
        self.len = np.array([len(d) for d in docs_tokens], dtype=float)
        self.avg = self.len.mean() if self.N else 0.0
        self.post = defaultdict(list)        # term -> list[(doc_id, tf)]
        df = defaultdict(int)
        for i, toks in enumerate(docs_tokens):
            tf = defaultdict(int)
            for t in toks: tf[t] += 1
            for t, c in tf.items():
                self.post[t].append((i, c)); df[t] += 1
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def scores(self, query_tokens):
        s = np.zeros(self.N)
        for t in query_tokens:
            if t not in self.post: continue
            idf = self.idf[t]
            for i, tf in self.post[t]:
                denom = tf + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
                s[i] += idf * tf * (self.k1 + 1) / denom
        return s

# ---------- Char-TFIDF (морфологія/одруки) ----------
class CharIndex:
    def __init__(self, texts):
        self.vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2)
        self.M = self.vec.fit_transform(texts)
    def scores(self, query):
        q = self.vec.transform([query])
        return linear_kernel(q, self.M).ravel()

# ---------- Dense-ембединги e5 (основний ранкер) ----------
class DenseE5Index:
    """Багатомовні ембединги intfloat/multilingual-e5. Модель тягнеться раз (~1.1 ГБ),
    ембединги 11k листів кешуються на диск — далі вантажаться з файлу за секунду.
        pip install sentence-transformers
    Кеш інвалідовується автоматично, якщо змінився корпус або назва моделі."""
    def __init__(self, texts, model_name='intfloat/multilingual-e5-base', cache_dir='.'):
        self.model_name = model_name
        self._model = None                                   # лінива загрузка
        fp = self._fingerprint(model_name, texts)
        self.npy  = os.path.join(cache_dir, 'uktzed_e5_cache.npy')
        self.meta = os.path.join(cache_dir, 'uktzed_e5_cache.meta.json')
        if self._cache_valid(fp):
            self.E = np.load(self.npy)
            print(f"  e5: ембединги з кешу ({self.E.shape[0]} листів)")
        else:
            print(f"  e5: рахую ембединги для {len(texts)} листів (раз)…")
            self.E = self._encode(['passage: ' + t for t in texts])
            np.save(self.npy, self.E)
            json.dump({'fp': fp, 'model': model_name}, open(self.meta, 'w', encoding='utf-8'))
            print("  e5: збережено в кеш")

    @staticmethod
    def _fingerprint(model_name, texts):
        h = hashlib.sha256(); h.update(model_name.encode('utf-8'))
        h.update(str(len(texts)).encode())
        for t in texts: h.update(t.encode('utf-8'))
        return h.hexdigest()

    def _cache_valid(self, fp):
        if not (os.path.exists(self.npy) and os.path.exists(self.meta)):
            return False
        try:
            return json.load(open(self.meta, encoding='utf-8')).get('fp') == fp
        except Exception:
            return False

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _encode(self, texts):
        return np.asarray(self.model.encode(texts, normalize_embeddings=True,
                                            show_progress_bar=True, batch_size=64))

    def scores(self, query):
        q = self.model.encode(['query: ' + query], normalize_embeddings=True)[0]
        return self.E @ q

# ---------- Злиття RRF ----------
def rrf(*score_arrays, k=60, weights=None, top_n=200):
    # Кожен ранкер нагороджує лише свій топ-N і лише документи зі score>0.
    # Це прибирає вироджений «пріоритет порядку індексів», коли ранкер порожній.
    n = len(score_arrays[0]); fused = np.zeros(n)
    weights = weights or [1.0] * len(score_arrays)
    for w, s in zip(weights, score_arrays):
        for rank, idx in enumerate(np.argsort(-s)[:top_n]):
            if s[idx] <= 0:        # масив відсортований за спаданням → далі лише нулі
                break
            fused[idx] += w / (k + rank + 1)
    return fused

# ---------- Ретривер ----------
class UktzedRetriever:
    def __init__(self, jsonl, use_dense=False):
        self.nodes = [json.loads(l) for l in open(jsonl, encoding='utf-8')]
        self.leaves = [n for n in self.nodes if n.get('is_leaf')]
        self.canon = {n['code']: n for n in self.nodes if n.get('code')}  # верифікація
        texts = [clean_chunk(n['chunk']) for n in self.leaves]
        self.bm25 = BM25([tokenize(t) for t in texts])
        self.char = CharIndex(texts)
        self.dense = None
        if use_dense:
            try:
                cache_dir = os.path.dirname(os.path.abspath(jsonl))
                self.dense = DenseE5Index(texts, cache_dir=cache_dir)
            except Exception as e:
                print(f"  e5 недоступний ({e.__class__.__name__}: {e}) — працюю на лексиці.")
                self.dense = None

    def search(self, query, top_k=20):
        arrays  = [self.bm25.scores(tokenize(query)), self.char.scores(query)]
        weights = [0.7, 0.4]                       # лексика — допоміжна
        if self.dense is not None:
            arrays.append(self.dense.scores(query))
            weights.append(1.3)                    # e5 — основний ранкер
        fused = rrf(*arrays, weights=weights)
        if not fused.any():          # жоден ранкер нічого не знайшов → не вгадуємо
            return []
        out = []
        for i in fused.argsort()[::-1][:top_k]:
            if fused[i] <= 0:
                break
            n = self.leaves[i]
            assert n['code'] in self.canon, f"код {n['code']} відсутній у наборі!"  # верифікація
            rec = self.canon[n['code']]
            out.append({'code': n['code'], 'name': n['name'], 'path': n['chunk'],
                        'rates': rec.get('rates'), 'ovo': rec.get('ovo'),
                        'score': round(float(fused[i]), 4)})
        return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('query', nargs='*')
    ap.add_argument('-i', '--interactive', action='store_true')
    ap.add_argument('-k', type=int, default=20)
    ap.add_argument('--dense', action='store_true', help='увімкнути e5 (потрібна модель)')
    ap.add_argument('--jsonl', default=os.path.join(os.path.dirname(__file__) or '.', 'uktzed_nodes.jsonl'))
    a = ap.parse_args()
    if not os.path.exists(a.jsonl):
        a.jsonl = '/mnt/user-data/outputs/uktzed_nodes.jsonl'
    r = UktzedRetriever(a.jsonl, use_dense=a.dense)
    print(f"Індекс: {len(r.leaves)} листів, {len(r.canon)} кодів усього"
          f"{' · e5 увімкнено' if r.dense else ' · лексика'}\n")

    def show(q):
        print(f"🔎 {q}")
        for h in r.search(q, a.k):
            print(f"  {h['code']}  [{h['score']}]  ставки={h['rates']}")
            print(f"      { h['path'] }")
        print()

    if a.interactive:
        while True:
            try: q = input('запит> ').strip()
            except (EOFError, KeyboardInterrupt): break
            if q: show(q)
    elif a.query:
        show(' '.join(a.query))
    else:
        for q in ["жіноча вовняна сукня", "ядерний реактор", "димогарний котел",
                  "чоловічі шкіряні черевики", "мобільний телефон смартфон",
                  "мелена кава", "зимові шини для легкового авто"]:
            show(q)

if __name__ == '__main__':
    main()
