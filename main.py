import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from database import create_document, get_documents, db
from schemas import Userprofile, Routine, Tip

app = FastAPI(title="LooksMax API", description="Healthy, habit-based appearance improvement API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "LooksMax Backend is running"}

@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    return response

# ------------------------
# Domain models and routes
# ------------------------

class CreateRoutineRequest(BaseModel):
    title: str
    steps: List[str]
    category: str
    owner_email: Optional[str] = None

@app.post("/api/routines")
def create_routine(payload: CreateRoutineRequest):
    routine = Routine(**payload.model_dump())
    routine_id = create_document("routine", routine)
    return {"id": routine_id, "message": "Routine created"}

@app.get("/api/routines")
def list_routines(owner_email: Optional[str] = None, category: Optional[str] = None, limit: int = 50):
    filters = {}
    if owner_email:
        filters["owner_email"] = owner_email
    if category:
        filters["category"] = category
    docs = get_documents("routine", filters, limit)
    for d in docs:
        d["_id"] = str(d.get("_id"))
    return docs

class CreateTipRequest(BaseModel):
    category: str
    title: str
    body: str
    tags: List[str] = []

@app.post("/api/tips")
def create_tip(payload: CreateTipRequest):
    tip = Tip(**payload.model_dump())
    tip_id = create_document("tip", tip)
    return {"id": tip_id, "message": "Tip created"}

@app.get("/api/tips")
def list_tips(category: Optional[str] = None, limit: int = 50):
    filters = {"category": category} if category else {}
    docs = get_documents("tip", filters, limit)
    for d in docs:
        d["_id"] = str(d.get("_id"))
    return docs

@app.post("/api/profile")
def upsert_profile(profile: Userprofile):
    # simple upsert by email if provided, else just create
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")
    data = profile.model_dump()
    from datetime import datetime, timezone
    data["updated_at"] = datetime.now(timezone.utc)
    if profile.email:
        result = db.userprofile.update_one({"email": profile.email}, {"$set": data}, upsert=True)
        doc = db.userprofile.find_one({"email": profile.email})
    else:
        inserted_id = db.userprofile.insert_one(data).inserted_id
        doc = db.userprofile.find_one({"_id": inserted_id})
    doc["_id"] = str(doc["_id"]) if doc.get("_id") else None
    return doc

@app.get("/api/profile")
def get_profile(email: str):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")
    doc = db.userprofile.find_one({"email": email})
    if not doc:
        raise HTTPException(status_code=404, detail="Profile not found")
    doc["_id"] = str(doc["_id"]) if doc.get("_id") else None
    return doc

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
