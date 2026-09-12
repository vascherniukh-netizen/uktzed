# -*- coding: utf-8 -*-
"""
Вебсервіс УКТ ЗЕД-агента: FastAPI поверх наявного ретривера й агента.

Ендпоінти:
  POST /classify   {session_id?, message}  →  {session_id, reply}
                   Підтримує багатоходовий діалог (уточнення) через session_id.
  GET  /           →  жива демо-сторінка (uktzed_demo_live.html, якщо поруч)
  GET  /health     →  {ok, dense, leaves}

Ретривер і ембединги піднімаються РАЗ при старті (з кешу — за секунду).

ЗАХИСТ ВІД ЗЛИВУ БЮДЖЕТУ (додано 12.09.2026).
Ендпоінт публічний і без авторизації, а кожен виклик коштує грошей у DeepSeek.
Тому чотири стелі, усі налаштовуються через змінні середовища:
  UKTZED_ALLOWED_ORIGINS  — кому можна звертатися з браузера (CORS).
                            Було "*" — будь-який сайт у світі міг крутити цей
                            ендпоінт зі сторінки відвідувача.
  UKTZED_RATE_LIMIT       — скільки запитів з однієї IP за вікно (типово 12).
  UKTZED_RATE_WINDOW      — довжина вікна в секундах (типово 600).
  UKTZED_MAX_CHARS        — стеля довжини повідомлення (типово 500).
                            Без неї будь-хто шле мегабайт тексту прямо в промпт.
  UKTZED_MAX_TURNS        — стеля реплік на одну сесію (типово 12).
                            Історія росте з кожним ходом і щоразу вирушає в API,
                            тож без стелі вартість однієї сесії необмежена.

session_id більше не можна вигадати: сервер приймає лише той ідентифікатор,
який сам раніше видав. Інакше сторонній міг би зайняти памʼять довільними
ключами або потрапити в чужу історію розмови.

Залежності: fastapi, uvicorn, openai (+ ретривер тягне sklearn/numpy, e5 — опційно).
    pip install fastapi "uvicorn[standard]" openai

LLM-провайдер: DeepSeek (OpenAI-сумісний API).

Запуск (Windows cmd):
    set DEEPSEEK_API_KEY=sk-...
    set UKTZED_DENSE=1            # увімкнути e5 (за наявності моделі+кешу)
    python -m uvicorn uktzed_api:app --host 0.0.0.0 --port 8000
"""
import os
import time
import uuid
from collections import deque

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from uktzed_retriever import UktzedRetriever
from uktzed_agent import UktzedAgent, MODEL

HERE = os.path.dirname(os.path.abspath(__file__))
JSONL = os.path.join(HERE, "uktzed_nodes.jsonl")
DENSE = os.environ.get("UKTZED_DENSE", "") not in ("", "0", "false", "False")
DEMO = os.path.join(HERE, "uktzed_demo_live.html")

# --- стелі ---------------------------------------------------------------
DEFAULT_ORIGINS = (
    "https://agents-that-don-t-invent-production.up.railway.app,"
    "http://localhost:3000,http://localhost:5173"
)
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get("UKTZED_ALLOWED_ORIGINS", DEFAULT_ORIGINS).split(",")
    if o.strip()
]
RATE_LIMIT = int(os.environ.get("UKTZED_RATE_LIMIT", 12))
RATE_WINDOW = int(os.environ.get("UKTZED_RATE_WINDOW", 600))
MAX_CHARS = int(os.environ.get("UKTZED_MAX_CHARS", 500))
MAX_TURNS = int(os.environ.get("UKTZED_MAX_TURNS", 12))
MAX_SESSIONS = int(os.environ.get("UKTZED_MAX_SESSIONS", 500))
LLM_TIMEOUT = float(os.environ.get("UKTZED_LLM_TIMEOUT", 60))

app = FastAPI(title="УКТ ЗЕД класифікатор")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)

STATE = {"retriever": None, "client": None, "sessions": {}, "turns": {}, "hits": {}}


def _client():
    from openai import OpenAI
    from uktzed_agent import BASE_URL

    key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("UKTZED_API_KEY")
    if not key:
        # Назовні — нейтральне формулювання: конфігурація сервера не є
        # публічною інформацією. Деталь лишається в логах.
        print("КОНФІГ: не задано DEEPSEEK_API_KEY")
        raise HTTPException(503, "Сервіс тимчасово недоступний")
    # Без таймауту зависла відповідь провайдера тримає воркер до нескінченності.
    return OpenAI(api_key=key, base_url=BASE_URL, timeout=LLM_TIMEOUT)


def _client_ip(request: Request) -> str:
    """За проксі Railway реальна адреса приходить у X-Forwarded-For."""
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _rate_ok(ip: str) -> bool:
    """Ковзне вікно на IP. Без сторонніх залежностей і без стану на диску."""
    now = time.time()
    hits = STATE["hits"].setdefault(ip, deque())
    while hits and now - hits[0] > RATE_WINDOW:
        hits.popleft()
    if len(hits) >= RATE_LIMIT:
        return False
    hits.append(now)
    # Щоб словник не ріс вічно від разових відвідувачів.
    if len(STATE["hits"]) > 5000:
        for k in [k for k, v in STATE["hits"].items() if not v or now - v[-1] > RATE_WINDOW][:1000]:
            STATE["hits"].pop(k, None)
    return True


@app.on_event("startup")
def _startup():
    print("Підняття ретривера…", "e5" if DENSE else "лексика")
    STATE["retriever"] = UktzedRetriever(JSONL, use_dense=DENSE)
    print(f"Готово: {len(STATE['retriever'].leaves)} листів.")
    print(f"CORS: {ALLOWED_ORIGINS}")
    print(f"Ліміти: {RATE_LIMIT}/{RATE_WINDOW}s на IP · {MAX_CHARS} символів · {MAX_TURNS} реплік на сесію")


class ClassifyIn(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, max_length=64)


@app.post("/classify")
def classify(inp: ClassifyIn, request: Request):
    ip = _client_ip(request)
    if not _rate_ok(ip):
        raise HTTPException(
            429,
            f"Забагато запитів. Спробуйте за {RATE_WINDOW // 60} хв — "
            "це демо з обмеженою квотою.",
        )

    msg = (inp.message or "").strip()
    if not msg:
        raise HTTPException(400, "Порожній запит")
    if len(msg) > MAX_CHARS:
        raise HTTPException(413, f"Опис задовгий: максимум {MAX_CHARS} символів")

    # Сесію відкриває тільки сервер. Невідомий ідентифікатор не створює нову
    # сесію під чужим ключем — натомість видається свіжий.
    sid = inp.session_id if inp.session_id in STATE["sessions"] else None
    if sid is None:
        sid = uuid.uuid4().hex
        if STATE["client"] is None:
            STATE["client"] = _client()
        STATE["sessions"][sid] = UktzedAgent(STATE["retriever"], STATE["client"])
        STATE["turns"][sid] = 0
        while len(STATE["sessions"]) > MAX_SESSIONS:
            old = next(iter(STATE["sessions"]))
            STATE["sessions"].pop(old, None)
            STATE["turns"].pop(old, None)

    if STATE["turns"].get(sid, 0) >= MAX_TURNS:
        raise HTTPException(
            429,
            f"Ліміт демо — {MAX_TURNS} реплік у розмові. Почніть нову, "
            "оновивши сторінку.",
        )

    STATE["turns"][sid] = STATE["turns"].get(sid, 0) + 1
    reply = STATE["sessions"][sid].turn(msg)
    return {"session_id": sid, "reply": reply}


@app.get("/health")
def health():
    r = STATE["retriever"]
    return {"ok": r is not None, "dense": bool(r and r.dense),
            "leaves": len(r.leaves) if r else 0, "model": MODEL}


@app.get("/")
def root():
    if os.path.exists(DEMO):
        return FileResponse(DEMO)
    return JSONResponse({"service": "УКТ ЗЕД класифікатор",
                         "endpoints": ["POST /classify", "GET /health"]})
