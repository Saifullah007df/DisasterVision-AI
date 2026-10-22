# DisasterVision AI

AI-powered image-based disaster detection using FastAPI and a pretrained CLIP vision-language model.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Open http://127.0.0.1:8000.

The first analysis downloads model weights. Predictions are experimental and must not be treated as emergency safety assessments.
