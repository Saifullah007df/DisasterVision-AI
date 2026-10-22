import io
import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from transformers import pipeline

BASE = Path(__file__).resolve().parent
DB = BASE / 'history.db'
MODEL_ID = os.getenv('DISASTER_MODEL', 'openai/clip-vit-base-patch32')
MAX_BYTES = 8 * 1024 * 1024
CATEGORIES = {
    'Flood': ['a photograph of a severe flood submerging streets and houses', 'a photograph of a river overflowing into a populated neighborhood'],
    'Earthquake damage': ['a photograph of buildings collapsed after an earthquake', 'a photograph of severe earthquake rubble and destroyed buildings'],
    'Landslide': ['a photograph of a landslide covering a road with mud and rocks', 'a photograph of a collapsed hillside and landslide debris'],
    'Wildfire': ['a photograph of a forest wildfire with flames and thick smoke', 'a photograph of a large vegetation fire burning trees'],
    'Tornado': ['a photograph of a tornado funnel touching the ground', 'a photograph of a violent tornado moving across a landscape'],
    'Cyclone or storm damage': ['a photograph of severe hurricane wind damage and fallen trees', 'a photograph of buildings damaged by a violent cyclone storm'],
    'Tsunami or coastal flooding': ['a photograph of a destructive tsunami wave inundating a coastal town', 'a photograph of severe coastal flooding with seawater engulfing buildings'],
    'No visible disaster': ['a photograph of a normal undamaged city street', 'a photograph of an ordinary peaceful residential neighborhood', 'a photograph of a healthy forest on a calm day', 'a photograph of an ordinary indoor room without damage'],
}
PROMPTS = [p for prompts in CATEGORIES.values() for p in prompts]
PROMPT_CATEGORY = {p: c for c, prompts in CATEGORIES.items() for p in prompts}
app = FastAPI(title='DisasterVision AI')
app.mount('/static', StaticFiles(directory=BASE / 'static'), name='static')
model = None
model_lock = threading.Lock()

def get_model():
    global model
    if model is None:
        with model_lock:
            if model is None:
                model = pipeline('zero-shot-image-classification', model=MODEL_ID)
    return model

def connection():
    con = sqlite3.connect(DB)
    con.execute('CREATE TABLE IF NOT EXISTS predictions (id TEXT PRIMARY KEY, created_at TEXT, filename TEXT, verdict TEXT, category TEXT, score REAL, rankings TEXT)')
    return con

@app.get('/', response_class=HTMLResponse)
def home():
    return (BASE / 'templates' / 'index.html').read_text(encoding='utf-8')

@app.get('/api/history')
def history():
    with connection() as con:
        rows = con.execute('SELECT id,created_at,filename,verdict,category,score FROM predictions ORDER BY created_at DESC LIMIT 30').fetchall()
    return [dict(zip(['id','created_at','filename','verdict','category','score'], r)) for r in rows]

@app.post('/api/analyze')
async def analyze(file: UploadFile = File(...)):
    if file.content_type not in ('image/jpeg', 'image/png', 'image/webp'):
        raise HTTPException(415, 'Upload a JPEG, PNG, or WEBP image.')
    raw = await file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, 'Image must be 8 MB or smaller.')
    try:
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw)))
        image = image.convert('RGB')
        if image.width * image.height > 25_000_000:
            raise ValueError('Image dimensions too large')
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(400, 'Invalid or excessively large image.')
    try:
        outputs = get_model()(image, candidate_labels=PROMPTS)
    except Exception as exc:
        raise HTTPException(503, 'Model unavailable. Check your internet connection for the first model download and server logs.') from exc
    # Relative image/text similarity scores are not calibrated probabilities.
    scores = {category: max(float(x['score']) for x in outputs if PROMPT_CATEGORY[x['label']] == category) for category in CATEGORIES}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    category, score = ranked[0]
    margin = score - ranked[1][1]
    uncertain = score < 0.22 or margin < 0.035
    verdict = 'UNCERTAIN' if uncertain else ('NO' if category == 'No visible disaster' else 'YES')
    explanation = ('Visual categories are too close to distinguish reliably.' if uncertain else
                   'The image most closely matches normal, undamaged scene descriptions.' if verdict == 'NO' else
                   f'The image most closely matches descriptions of {category.lower()}. This does not verify the event or its cause.')
    record = {'id': str(uuid.uuid4()), 'created_at': datetime.now(timezone.utc).isoformat(),
              'filename': Path(file.filename or 'upload').name[:120], 'verdict': verdict,
              'category': 'Uncertain' if uncertain else category, 'score': round(score, 4),
              'rankings': [{'category': name, 'similarity': round(value, 4)} for name, value in ranked],
              'explanation': explanation,
              'disclaimer': 'Image-only classification; not a forecast, emergency alert, or safety determination. Similarity scores are not calibrated confidence percentages.'}
    with connection() as con:
        con.execute('INSERT INTO predictions VALUES (?,?,?,?,?,?,?)', (record['id'], record['created_at'], record['filename'], verdict, record['category'], score, json.dumps(record['rankings'])))
    return record
