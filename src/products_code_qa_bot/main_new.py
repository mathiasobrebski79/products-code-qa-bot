"""
Serveur local exposant le QA bot (TAPAS) sur ta table CSV.
Accessible depuis n'importe quel appareil du même réseau WiFi.

Installation :
    uv add fastapi "uvicorn[standard]"

Lancement :
    uvicorn main:app --host 0.0.0.0 --port 8000

Puis, depuis le téléphone (même WiFi) :
    http://<IP_DE_TON_PC>:8000
"""

import pandas as pd
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import pipeline

from settings import DATA_DIR  # adapte selon la structure de ton projet

app = FastAPI(title="HACCP QA Bot")

# Chargés une seule fois au démarrage du serveur, pas à chaque question
qa_bot = pipeline(task="table-question-answering", model="google/tapas-small-finetuned-sqa")
table = pd.read_csv(DATA_DIR / "codes.csv").astype(str)  # adapte le nom du fichier


class Question(BaseModel):
    question: str


@app.post("/ask")
def ask(payload: Question):
    result = qa_bot(table=table, query=payload.question)
    return {
        "answer": result["answer"],
        "cells": result.get("cells", []),
    }


@app.get("/health")
def health():
    """Utile pour vérifier rapidement depuis le téléphone que le serveur répond."""
    return {"status": "ok", "rows": len(table)}


# Sert la page web (index.html) à la racine
app.mount("/", StaticFiles(directory="static", html=True), name="static")
