from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import os
import io
from pathlib import Path

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️  Pillow not installed. Install with: pip install Pillow")

app = FastAPI(title="CareMaa ML Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Risk Model ────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "model" / "model.pkl"
model = None

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("✅ Risk ML model loaded")
except FileNotFoundError:
    print("⚠️  model.pkl not found. Run train.py first. Using rule-based fallback.")


# ── Risk Prediction ────────────────────────────────────────────────────────────
class RiskInput(BaseModel):
    age: float
    systolicBP: float
    diastolicBP: float
    bloodSugar: float
    bodyTemp: float
    heartRate: float


def rule_based_risk(data: RiskInput):
    score = 0
    conditions = []

    if data.systolicBP >= 140 or data.diastolicBP >= 90:
        score += 3
        conditions.append("Preeclampsia risk (High BP)")
    elif data.systolicBP >= 130:
        score += 1
        conditions.append("Mild Hypertension")

    if data.bloodSugar >= 140:
        score += 2
        conditions.append("Gestational Diabetes risk")
    elif data.bloodSugar >= 120:
        score += 1
        conditions.append("Borderline Blood Sugar")

    if data.heartRate > 100:
        score += 1
        conditions.append("Tachycardia")

    if data.bodyTemp > 99.5:
        score += 1
        conditions.append("Fever")

    if data.age > 35:
        score += 1
        conditions.append("Advanced Maternal Age")

    risk_level = "High" if score >= 4 else "Medium" if score >= 2 else "Low"
    confidence = round(min(0.97, 0.60 + score * 0.06), 2)
    detected = ", ".join(conditions) if conditions else "No significant risk factors"

    recs = []
    if risk_level == "High":
        recs += ["🚨 Seek immediate medical attention", "📞 Call your OB/GYN now", "🏥 Consider emergency visit"]
    elif risk_level == "Medium":
        recs += ["📅 Schedule check-up within 48 hours", "📊 Monitor vitals daily", "😴 Rest adequately"]
    else:
        recs += ["✅ Continue routine prenatal care", "💧 Stay hydrated", "🚶 Light exercise is safe"]

    if data.systolicBP >= 140:
        recs.append("🧘 Practice stress-reduction techniques")
    if data.bloodSugar >= 140:
        recs.append("🥗 Follow gestational diabetes diet")

    return {
        "riskLevel": risk_level,
        "confidence": confidence,
        "detectedCondition": detected,
        "recommendations": recs,
        "rawScore": score,
    }


@app.post("/predict-risk")
async def predict_risk(data: RiskInput):
    try:
        if model is not None:
            features = np.array([[
                data.age, data.systolicBP, data.diastolicBP,
                data.bloodSugar, data.bodyTemp, data.heartRate
            ]])
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0]
            confidence = round(float(max(probability)), 2)
            risk_map = {0: "Low", 1: "Medium", 2: "High"}
            risk_level = risk_map.get(int(prediction), "Medium")
            rule_result = rule_based_risk(data)
            return {
                "riskLevel": risk_level,
                "confidence": confidence,
                "detectedCondition": rule_result["detectedCondition"],
                "recommendations": rule_result["recommendations"],
                "source": "ml_model",
            }
        else:
            result = rule_based_risk(data)
            result["source"] = "rule_based"
            return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Anemia Detection ───────────────────────────────────────────────────────────
def analyze_pallor(image, scan_type: str):
    """
    RGB channel-based pallor analysis.
    Healthy conjunctiva / nail beds → high R relative to G+B (pinkish-red).
    Anemic tissue → reduced R ratio, elevated G ratio (pale/yellowy).
    Pallor score 0-100: higher = more pale = higher anemia risk.
    """
    img = image.resize((200, 200)).convert("RGB")
    pixels = np.array(img).reshape(-1, 3).astype(float)

    r_mean = float(np.mean(pixels[:, 0]))
    g_mean = float(np.mean(pixels[:, 1]))
    b_mean = float(np.mean(pixels[:, 2]))

    total = r_mean + g_mean + b_mean + 1e-6
    r_ratio = r_mean / total
    g_ratio = g_mean / total

    pallor_score = round((1.0 - r_ratio + g_ratio) * 50.0, 1)
    pallor_score = max(0.0, min(100.0, pallor_score))

    if pallor_score >= 65:
        risk = "High"
        hb_cat = "Possibly < 8 g/dL (Severe Anemia)"
        rec = "Strong pallor detected. Please visit a clinic for a CBC blood test immediately. 🚨"
        confidence = round(min(0.95, 0.75 + (pallor_score - 65) * 0.005), 2)
    elif pallor_score >= 50:
        risk = "Medium"
        hb_cat = "Possibly 8-11 g/dL (Mild-Moderate Anemia)"
        rec = "Moderate pallor detected. Consider iron-rich foods and consult a doctor for a blood test. ⚠️"
        confidence = round(min(0.95, 0.65 + (pallor_score - 50) * 0.005), 2)
    else:
        risk = "Low"
        hb_cat = "Likely >= 11 g/dL (Normal Range)"
        rec = "No significant pallor detected. Maintain a balanced diet with iron and vitamin C. ✅"
        confidence = round(min(0.95, 0.60 + (50 - pallor_score) * 0.004), 2)

    return {
        "anemiaRisk": risk,
        "confidence": confidence,
        "estimatedHemoglobinCategory": hb_cat,
        "recommendation": rec,
        "palorScore": pallor_score,
        "rValue": round(r_mean, 1),
        "gValue": round(g_mean, 1),
        "bValue": round(b_mean, 1),
    }


@app.post("/predict-anemia")
async def predict_anemia(image: UploadFile = File(...), scan_type: str = Form("eyelid")):
    try:
        if not PIL_AVAILABLE:
            return {
                "anemiaRisk": "Medium",
                "confidence": 0.62,
                "estimatedHemoglobinCategory": "Possibly 8-11 g/dL (install Pillow for full analysis)",
                "recommendation": "Run: pip install Pillow  in the ml-service directory for full analysis.",
                "palorScore": 52.0,
                "rValue": 130.0,
                "gValue": 110.0,
                "bValue": 100.0,
            }
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))
        return analyze_pallor(img, scan_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Anemia analysis failed: {str(e)}")


# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "CareMaa ML Service running 🌸",
        "model_loaded": model is not None,
        "pil_available": PIL_AVAILABLE,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
