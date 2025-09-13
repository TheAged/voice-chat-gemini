from fastapi import APIRouter
from app.services.fall_detection_service import fall_warning

router = APIRouter()

@router.get("/fall_status")
def get_fall_status():
    return {"status": fall_warning}
