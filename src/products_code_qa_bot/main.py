from pathlib import Path

from transformers import pipeline
import pandas as pd
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from products_code_qa_bot.settings import DATA_DIR

# def main():
    

app = FastAPI(title="Meat products codes QA Bot")

code_table = pd.read_csv("data/codes.csv").astype(str)

# google/tapas-base-finetuned-wtq

qa_bot = pipeline(
    task="table-question-answering",
    model="google/tapas-small-finetuned-sqa"
    )


class Question(BaseModel):
    question: str


@app.post("/ask")
def ask(payload: Question):
    result = qa_bot(
        table=code_table, 
        query=payload.question
        )
    return {
        "answer": result["answer"],
        "cells": result.get("cells", []),
    }


@app.get("/health")
def health():
    """Utile pour vérifier rapidement depuis le téléphone que le serveur répond."""
    return {"status": "ok", "rows": len(code_table)}


# Sert la page web (index.html) à la racine

STATIC_DIR = Path(__file__).resolve().parent / "static"

app.mount(
    "/", 
    StaticFiles(directory=STATIC_DIR, html=True), 
    name="static")

    # print("Hello from products-code-qa-bot! Type exit to stop")
    
    # while True :

    #     question = input("Ask a question (or type 'exit'): ")

    #     if question.lower() == "exit":
    #         break
    
    #     result = qa_bot(
    #         query=question, 
    #         table=code_table
    #         )
        
    #     print(f"Answer: {result['answer']}\n")
    


# if __name__ == "__main__":
#     main()
