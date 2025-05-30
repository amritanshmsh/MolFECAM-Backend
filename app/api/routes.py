from fastapi import APIRouter
from app.schemas import MoleculeRequest, PredictionResponse
# from app.services.predictor import make_prediction
from app.models.molformer import initialize_pipeline, predict_properties
router = APIRouter()

feature_extractor, classifier = initialize_pipeline()

@router.post("/predict", response_model=list[PredictionResponse])
def predict_molecular_properties(request: MoleculeRequest):
    # Validate request
    if not request.smiles:
        return {"error": "No SMILES provided", "results": []}
    
    # Filter out empty strings
    valid_smiles = [s for s in request.smiles if s and s.strip()]
    
    if not valid_smiles:
        return {"error": "No valid SMILES provided", "results": []}
    
    predictions = predict_properties(valid_smiles)
    return {"results": predictions}
