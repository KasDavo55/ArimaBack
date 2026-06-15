from fastapi import APIRouter, HTTPException
from app.models.schemas import RecommendRequest, RecommendResponse
from app.services.recommend_service import recommend_model

router = APIRouter(prefix="/recommend-model", tags=["recommend"])


@router.post("", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    try:
        result = recommend_model(req.summary)
        return result
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error en recomendación: {e}")