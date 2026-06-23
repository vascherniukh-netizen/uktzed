# -*- coding: utf-8 -*-
"""
Вебсервіс УКТ ЗЕД-агента: FastAPI поверх наявного ретривера й агента.

Ендпоінти:
  POST /classify   {session_id?, message}  →  {session_id, reply}
                   Підтримує багатоходовий діалог (уточнення) через session_id.
  GET  /           →  жива демо-сторінка (uktzed_demo_live.html, якщо поруч)
  GET  /health     →  {ok, dense, leaves}

Ретривер і ембединги піднімаються РАЗ при старті (з кешу — за секунду).
Кожна сесія — окрема історія розмови; інстанси тримаються в памʼяті
(для тесту з друзями достатньо; для продакшну — винести у сховище).

Залежності: fastapi, uvicorn, anthropic (+ ретривер тягне sklearn/numpy, e5 — опційно).
    pip install fastapi "uvicorn[standard]" anthropic

Запуск (Windows cmd):
    set ANTHROPIC_API_KEY=sk-ant-...
    set UKTZED_DENSE=1            # увімкнути e5 (за наявності моделі+кешу)
    python -m uvicorn uktzed_api:app --host 0.0.0.0 --port 8000
"""
import os, uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from uktzed_retriever import UktzedRetriever
from uktzed_agent import UktzedAgent

HERE = os.path.dirname(os.path.abspath(__file__))
JSONL = os.path.join(HERE, "uktzed_nodes.jsonl")
DENSE = os.environ.get("UKTZED_DENSE", "") not in ("", "0", "false", "False")
DEMO = os.path.join(HERE, "uktzed_demo_live.html")

app = FastAPI(title="УКТ ЗЕД класифікатор")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

STATE = {"retriever": None, "client": None, "sessions": {}}


def _client():
    from anthropic import Anthropic
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise HTTPException(500, "На сервері не задано ANTHROPIC_API_KEY")
    return Anthropic(api_key=key)


@app.on_event("startup")
def _startup():
    print("Підняття ретривера…", "e5" if DENSE else "лексика")
    STATE["retriever"] = UktzedRetriever(JSONL, use_dense=DENSE)
    print(f"Готово: {len(STATE['retriever'].leaves)} листів.")


class ClassifyIn(BaseModel):
    message: str
    session_id: str | None = None


@app.post("/classify")
def classify(inp: ClassifyIn):
    msg = (inp.message or "").strip()
    if not msg:
        raise HTTPException(400, "Порожній запит")
    sid = inp.session_id or uuid.uuid4().hex
    agent = STATE["sessions"].get(sid)
    if agent is None:
        if STATE["client"] is None:
            STATE["client"] = _client()
        agent = UktzedAgent(STATE["retriever"], STATE["client"])
        STATE["sessions"][sid] = agent
        # обмежимо памʼять для тесту: не більше 500 живих сесій
        if len(STATE["sessions"]) > 500:
            STATE["sessions"].pop(next(iter(STATE["sessions"])))
    reply = agent.turn(msg)
    return {"session_id": sid, "reply": reply}


@app.get("/health")
def health():
    r = STATE["retriever"]
    return {"ok": r is not None, "dense": bool(r and r.dense), "leaves": len(r.leaves) if r else 0}


@app.get("/")
def root():
    if os.path.exists(DEMO):
        return FileResponse(DEMO)
    return JSONResponse({"service": "УКТ ЗЕД класифікатор",
                         "endpoints": ["POST /classify", "GET /health"]})
