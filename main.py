import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional

from database import create_document, get_documents, db
from schemas import Userprofile, Routine, Tip

# Optional heavy imports (cv2, numpy) will be available after requirements install
import numpy as np  # type: ignore
import cv2  # type: ignore

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
        db.userprofile.update_one({"email": profile.email}, {"$set": data}, upsert=True)
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

# ------------------------
# Face analysis endpoint
# ------------------------

def _safe_read_image(file_bytes: bytes):
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image format")
    return img


def _estimate_skin_tone(bgr_img, face_box):
    x, y, w, h = face_box
    h_img, w_img = bgr_img.shape[:2]
    x = max(0, x); y = max(0, y)
    w = max(1, w); h = max(1, h)
    x2 = min(w_img, x + w)
    y2 = min(h_img, y + h)
    roi = bgr_img[y:y2, x:x2]
    if roi.size == 0:
        roi = bgr_img

    # Blur to reduce noise, then convert to LAB (OpenCV range: L[0..255], a,b[0..255] centered at 128)
    roi_blur = cv2.GaussianBlur(roi, (5,5), 0)
    lab = cv2.cvtColor(roi_blur, cv2.COLOR_BGR2LAB)
    mean_lab = lab.reshape(-1, 3).mean(axis=0)
    L = float(mean_lab[0])
    a = float(mean_lab[1]) - 128.0
    b = float(mean_lab[2]) - 128.0

    # Level classification by lightness
    level = 'light'
    if L < 95:
        level = 'deep'
    elif 95 <= L < 160:
        level = 'medium'
    else:
        level = 'light'

    # Undertone heuristic
    if b > 8 and a > 4:
        undertone = 'warm'
    elif b < -6 and a < -2:
        undertone = 'cool'
    else:
        undertone = 'neutral'

    # Palette suggestions
    palettes = {
        'warm': [
            'earthy browns', 'olive', 'mustard', 'warm beige', 'rust', 'forest green'
        ],
        'cool': [
            'navy', 'charcoal', 'emerald', 'sapphire', 'cool gray', 'berry tones'
        ],
        'neutral': [
            'soft white', 'taupe', 'camel', 'teal', 'true red', 'mid-gray'
        ]
    }
    neutrals = {
        'deep': ['rich charcoal', 'espresso', 'cream'],
        'medium': ['navy', 'camel', 'off-white'],
        'light': ['light gray', 'sand', 'soft white']
    }

    return {
        'level': level,
        'undertone': undertone,
        'suggested_palette': palettes[undertone],
        'neutrals': neutrals[level],
        'metrics': {'L': round(L,1), 'a': round(a,1), 'b': round(b,1)}
    }


@app.post("/api/face/analyze")
async def analyze_face(file: UploadFile = File(...)):
    try:
        content = await file.read()
        img = _safe_read_image(content)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Face detection
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except Exception:
            face_cascade = None

        faces = []
        if face_cascade is not None:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        if len(faces) == 0:
            # fallback to using the central region if no face found
            h, w = gray.shape
            cx1, cy1 = int(w*0.2), int(h*0.2)
            cx2, cy2 = int(w*0.8), int(h*0.8)
            roi = gray[cy1:cy2, cx1:cx2]
            face_box = (cx1, cy1, cx2-cx1, cy2-cy1)
        else:
            # pick the largest face
            x, y, w0, h0 = max(faces, key=lambda f: f[2]*f[3])
            roi = gray[y:y+h0, x:x+w0]
            face_box = (int(x), int(y), int(w0), int(h0))

        roi_resized = cv2.resize(roi, (256, 256), interpolation=cv2.INTER_AREA)

        # Symmetry score (0-1): compare left vs mirrored right
        left = roi_resized[:, :128]
        right = roi_resized[:, 128:]
        right_flipped = cv2.flip(right, 1)
        diff = cv2.absdiff(left, right_flipped)
        mean_diff = float(np.mean(diff))
        symmetry_score = max(0.0, 1.0 - (mean_diff / 255.0) * 2.0)  # rough normalization

        # Edge/angleness via Laplacian variance
        lap = cv2.Laplacian(roi_resized, cv2.CV_64F)
        lap_var = float(lap.var())
        edge_score = min(1.0, lap_var / 5000.0)  # normalize roughly

        # Aspect ratio from face box
        w_face = face_box[2]
        h_face = face_box[3]
        aspect = w_face / max(1.0, float(h_face))

        # Heuristic shape classification
        shape = "oval"
        if aspect < 0.8:
            shape = "oblong"
        elif 0.8 <= aspect <= 1.05:
            shape = "round" if edge_score < 0.35 else "square"
        elif 1.05 < aspect <= 1.25:
            shape = "oval"
        else:
            shape = "heart" if symmetry_score > 0.6 else "diamond"

        # Simple aesthetic rating 1-10 using symmetry and proportionality proxy
        proportion_score = 1.0 - abs(aspect - 1.0)  # peak at 1.0 when aspect==1.0
        proportion_score = max(0.0, min(1.0, proportion_score))
        base = 4.5
        rating = base + 3.0*symmetry_score + 2.0*proportion_score + 0.5*(1.0 - edge_score)
        rating = float(max(1.0, min(10.0, rating)))

        # Skin tone estimation and palette
        tone = _estimate_skin_tone(img, face_box)

        # Guidance per shape
        shape_tips_map = {
            'oval': [
                'Most hairstyles suit you—experiment with texture or volume.',
                'Glasses: try rectangular or geometric frames to add angles.',
                'Facial hair: light stubble or short boxed styles keep balance.'
            ],
            'round': [
                'Add height and structure—short sides, volume on top.',
                'Glasses: angular frames (rectangle, D-frame).',
                'Facial hair: defined lines and longer goatee can elongate.'
            ],
            'square': [
                'Soften angles with slight texture or medium length.',
                'Glasses: round/oval frames to balance strong jaw.',
                'Facial hair: stubble or rounded edges to soften.'
            ],
            'oblong': [
                'Avoid extra height—add width with side volume or layers.',
                'Glasses: taller lenses, avoid narrow frames.',
                'Facial hair: fuller mustache/beard can reduce verticality.'
            ],
            'heart': [
                'Balance wider forehead with medium length and side volume.',
                'Glasses: light-colored or rimless lower visual weight.',
                'Facial hair: short boxed beards add weight to jawline.'
            ],
            'diamond': [
                'Add width at forehead/jaw; avoid too narrow sides.',
                'Glasses: oval frames; avoid very narrow bridges.',
                'Facial hair: short stubble to enhance jaw without sharpness.'
            ]
        }

        guidance = {
            'shape_tips': shape_tips_map.get(shape, []),
            'color_palette': tone['suggested_palette'],
            'neutrals': tone['neutrals']
        }

        return {
            "shape": shape,
            "symmetry": round(symmetry_score, 3),
            "proportion": round(proportion_score, 3),
            "edge": round(edge_score, 3),
            "aspect_ratio": round(aspect, 3),
            "rating": round(rating, 1),
            "box": {"x": face_box[0], "y": face_box[1], "w": face_box[2], "h": face_box[3]},
            "tone": tone,
            "guidance": guidance,
            "note": "Heuristic, for fun and educational use only"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)[:120]}")

# ------------------------
# Lightweight built-in UI (alt deployment)
# ------------------------
@app.get("/ui", response_class=HTMLResponse)
def lightweight_ui():
    # Single-file UI that calls the same-origin API endpoints
    html = """
    <!doctype html>
    <html>
      <head>
        <meta charset=\"utf-8\" />
        <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
        <title>LooksMax (Lite UI)</title>
        <style>
          body{font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin:0; background:#0b1220; color:#eef2ff}
          .wrap{max-width:960px;margin:0 auto;padding:24px}
          .card{background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.12);border-radius:12px;padding:16px;margin-top:16px}
          label{display:block;font-size:12px;color:#cbd5e1;margin:6px 0}
          input, textarea, select{width:100%;padding:8px 10px;border-radius:8px;border:1px solid rgba(255,255,255,0.15);background:rgba(255,255,255,0.08);color:#fff}
          button{background:#38bdf8;color:#082f49;border:0;border-radius:8px;padding:10px 14px;font-weight:600;cursor:pointer}
          button:disabled{opacity:0.6;cursor:not-allowed}
          .grid{display:grid;gap:12px}
          .row{display:flex;gap:8px;flex-wrap:wrap}
          small{color:#94a3b8}
          .pill{display:inline-block;border:1px solid rgba(255,255,255,0.15);padding:2px 8px;border-radius:999px;margin:2px;font-size:12px;color:#e2e8f0}
          img{max-height:220px;border-radius:10px;border:1px solid rgba(255,255,255,0.12)}
          pre{white-space:pre-wrap;background:rgba(15,23,42,0.7);padding:8px;border-radius:8px}
        </style>
      </head>
      <body>
        <div class=\"wrap\">
          <h1>LooksMax (Lite UI)</h1>
          <p><small>Single-page fallback hosted by the API. Uses same-origin requests.</small></p>

          <div class=\"card\">
            <h3>System Check</h3>
            <div id=\"status\">Checking...</div>
          </div>

          <div class=\"card\">
            <h3>Quick Profile</h3>
            <div class=\"grid\">
              <div><label>Name</label><input id=\"name\" placeholder=\"Optional\" /></div>
              <div><label>Email</label><input id=\"email\" placeholder=\"name@example.com\" /></div>
              <div><label>Skin type</label>
                <select id=\"skin\"><option></option><option>normal</option><option>oily</option><option>dry</option><option>combination</option><option>sensitive</option></select>
              </div>
              <div><label>Hair type</label>
                <select id=\"hair\"><option></option><option>straight</option><option>wavy</option><option>curly</option><option>coily</option></select>
              </div>
              <div><label>Style vibe</label>
                <select id=\"vibe\"><option></option><option>classic</option><option>minimal</option><option>streetwear</option><option>preppy</option><option>sporty</option></select>
              </div>
              <div class=\"row\"><button id=\"save\">Save Profile</button><div id=\"pstatus\"></div></div>
            </div>
          </div>

          <div class=\"card\">
            <h3>Routines</h3>
            <div class=\"row\">
              <select id=\"cat\">
                <option value=\"skin\">Skin</option>
                <option value=\"hair\">Hair</option>
                <option value=\"style\">Style</option>
                <option value=\"fitness\">Fitness</option>
                <option value=\"sleep\">Sleep</option>
                <option value=\"confidence\">Confidence</option>
              </select>
              <button id=\"refresh\">Refresh</button>
            </div>
            <div id=\"routines\" style=\"margin-top:6px\"></div>
            <div class=\"grid\" style=\"margin-top:8px\">
              <input id=\"rtitle\" placeholder=\"Routine title\" />
              <textarea id=\"rsteps\" placeholder=\"One step per line\" rows=\"4\"></textarea>
              <button id=\"addRoutine\">Create routine</button>
            </div>
          </div>

          <div class=\"card\">
            <h3>Face Analysis</h3>
            <input type=\"file\" id=\"file\" accept=\"image/*\" />
            <div class=\"row\" style=\"margin-top:8px\"><button id=\"analyze\">Analyze Photo</button><div id=\"anstatus\"></div></div>
            <div id=\"preview\" style=\"margin-top:8px\"></div>
            <pre id=\"analysis\" style=\"margin-top:8px\"></pre>
          </div>
        </div>
        <script>
          const qs = (s)=>document.querySelector(s);
          const statusEl = qs('#status');
          fetch('/')
            .then(r=>r.json()).then(d=>{statusEl.textContent = 'Backend: ' + (d.message || 'OK');})
            .then(()=>fetch('/test')).then(r=>r.json()).then(d=>{statusEl.textContent += ' | DB: ' + (d.database || 'n/a');})
            .catch(e=>{statusEl.textContent='Check failed: '+e.message});

          qs('#save').onclick = async ()=>{
            const body = {
              name: qs('#name').value,
              email: qs('#email').value,
              skin_type: qs('#skin').value,
              hair_type: qs('#hair').value,
              style_vibe: qs('#vibe').value,
            };
            qs('#pstatus').textContent = 'Saving...';
            try {
              const res = await fetch('/api/profile',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
              const data = await res.json();
              qs('#pstatus').textContent = res.ok ? 'Saved ✓' : (data.detail||'Failed');
            } catch(e){ qs('#pstatus').textContent = 'Error: '+e.message }
          };

          async function loadRoutines(){
            const cat = qs('#cat').value;
            const res = await fetch('/api/routines?category='+encodeURIComponent(cat));
            const arr = await res.json();
            const wrap = qs('#routines');
            wrap.innerHTML = '';
            (Array.isArray(arr)?arr:[]).forEach(r=>{
              const div = document.createElement('div');
              div.className='pill';
              div.textContent = r.title || 'Untitled';
              wrap.appendChild(div);
            })
          }
          qs('#refresh').onclick = loadRoutines;
          loadRoutines();

          qs('#addRoutine').onclick = async ()=>{
            const title = qs('#rtitle').value.trim();
            const stepsText = qs('#rsteps').value;
            if(!title) return;
            const steps = stepsText.split('\n').map(s=>s.trim()).filter(Boolean);
            const cat = qs('#cat').value;
            await fetch('/api/routines',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({title,steps,category:cat})});
            qs('#rtitle').value=''; qs('#rsteps').value='';
            loadRoutines();
          };

          const fileInput = qs('#file');
          fileInput.addEventListener('change',()=>{
            const f = fileInput.files && fileInput.files[0];
            const p = qs('#preview');
            p.innerHTML='';
            if(f){
              const url = URL.createObjectURL(f);
              const img = document.createElement('img');
              img.src = url; p.appendChild(img);
            }
          });

          qs('#analyze').onclick = async ()=>{
            const f = fileInput.files && fileInput.files[0];
            if(!f){ qs('#anstatus').textContent='Choose a photo first'; return; }
            qs('#anstatus').textContent='Analyzing...';
            const fd = new FormData(); fd.append('file', f);
            try {
              const res = await fetch('/api/face/analyze',{method:'POST', body: fd});
              const data = await res.json();
              qs('#anstatus').textContent = res.ok ? 'Done' : (data.detail||'Failed');
              qs('#analysis').textContent = JSON.stringify(data, null, 2);
            } catch(e){ qs('#anstatus').textContent = 'Error: '+e.message }
          };
        </script>
      </body>
    </html>
    """
    return HTMLResponse(content=html)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
