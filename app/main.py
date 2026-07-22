from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import os
from agent.inference import DQNAgent

app = FastAPI(title="Lux AI Agent API")

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model_final.h5")
agent = DQNAgent(MODEL_PATH)

class Observation(BaseModel):
    step: int
    player: int
    updates: List[str]
    class Config:
        extra = "allow"

import traceback

@app.post("/predict")
def predict(obs: Observation):
    try:
        actions = agent.act(obs.dict())
        return {"actions": actions}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
