# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from app.models.molformer import initialize_pipeline, predict_properties, get_forgetting_measures

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update to specific origin like ["http://localhost:3000"] if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if __name__ == "__main__":
    initialize_pipeline()

@app.get("/")
def read_root(): 
    return {"message": "FastAPI is running 🚀"}

# Prediction endpoint input schema
class MoleculeRequest(BaseModel):
    smiles: List[str]

@app.post("/predict")
def predict_molecular_properties(request: MoleculeRequest):
    predictions = predict_properties(request.smiles)
    return {"results": predictions}
@app.get("/metrics/forgetting")
def forgetting_metrics():
    return get_forgetting_measures()