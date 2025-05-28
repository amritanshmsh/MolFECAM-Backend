from pydantic import BaseModel
from typing import List

class SMILESRequest(BaseModel):
    smiles: List[str]

class PredictionResult(BaseModel):
    property: str
    value: str
    confidence: float

class PredictionResponse(BaseModel):
    predictions: List[PredictionResult]
