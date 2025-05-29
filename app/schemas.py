from pydantic import BaseModel, field_validator
from typing import List

class SMILESRequest(BaseModel):
    smiles: List[str]

class PredictionResult(BaseModel):
    property: str
    value: str
    confidence: float

class PredictionResponse(BaseModel):
    predictions: List[PredictionResult]

class SMILESInput(BaseModel):
    smiles: list[str]

class MoleculeRequest(BaseModel):
    smiles: List[str]
    
    @field_validator('smiles')
    def validate_smiles_list(cls, v):
        if not v:
            raise ValueError('SMILES list cannot be empty')
        
        # Check for empty strings
        empty_count = sum(1 for s in v if not s or not s.strip())
        if empty_count > 0:
            raise ValueError(f'Found {empty_count} empty SMILES strings')
            
        return v