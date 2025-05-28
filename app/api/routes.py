from fastapi import APIRouter
from app.schemas import SMILESInput, PredictionResponse
from app.services.predictor import make_prediction
from app.models.molformer import initialize_pipeline, predict_properties

router = APIRouter()

feature_extractor, classifier = initialize_pipeline()

@router.post("/predict", response_model=list[PredictionResponse])
def predict_molecular_properties(data: SMILESInput):
    results = predict_properties(data.smiles)
    return results
