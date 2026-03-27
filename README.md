# 🌸 CareMaa — AI-Powered Women's Health Platform

> A full-stack MERN + Python/FastAPI application with AI face verification, period tracking, pregnancy monitoring, vaccine scheduling, nutrition planning, community support, ML-based risk assessment, and an agentic AI assistant.

---

## 📁 Project Structure

```
caremaa-monorepo/
├── frontend/          React.js + Tailwind-style CSS
├── backend/           Node.js + Express.js + MongoDB
└── ml-service/        Python FastAPI + RandomForest ML Model
```

---

## ⚡ Quick Start

### 1. Prerequisites
- Node.js v18+
- Python 3.10+
- MongoDB Atlas URI (or local MongoDB)

---

### 2. Backend Setup

```bash
cd backend
npm install
```

Edit `.env`:
```
MONGO_URI=mongodb+srv://<user>:<pass>@cluster.mongodb.net/caremaa
JWT_SECRET=your_secret_key_here
CLIENT_URL=http://localhost:3000
ML_SERVICE_URL=http://127.0.0.1:8000
OPENAI_API_KEY=sk-...   # Optional — enables real GPT responses
```

```bash
npm run dev      # Starts on http://localhost:5000
```

---

### 3. ML Service Setup

```bash
cd ml-service
pip install -r requirements.txt

# Train the model (generates model/model.pkl)
python train.py

# Start the FastAPI server
uvicorn app:app --reload --port 8000
```

> ℹ️ If `model.pkl` is not found, the service automatically falls back to rule-based risk logic.

---

### 4. Frontend Setup

```bash
cd frontend
npm install
npm start    # Starts on http://localhost:3000
```

---

## 🔑 Environment Variables Summary

| File | Variable | Description |
|------|----------|-------------|
| `backend/.env` | `MONGO_URI` | **Required.** Your MongoDB connection string |
| `backend/.env` | `JWT_SECRET` | JWT signing secret |
| `backend/.env` | `OPENAI_API_KEY` | Optional. Enables real LLM responses in AI Assistant |
| `frontend/.env` | `REACT_APP_API_URL` | Backend API base URL |

---

## 🌐 API Reference

### Auth
| Method | Route | Description |
|--------|-------|-------------|
| POST | `/api/auth/register` | Register new user |
| POST | `/api/auth/login` | Login and get JWT |
| GET  | `/api/auth/me` | Get current user (protected) |

### Profile
| Method | Route | Description |
|--------|-------|-------------|
| GET  | `/api/profile` | Get user profile |
| PUT  | `/api/profile/setup` | Setup profile (first time) |
| PUT  | `/api/profile/update` | Update profile |

### Period Tracker
| Method | Route | Description |
|--------|-------|-------------|
| POST | `/api/cycle/calculate` | Calculate cycle dates |
| POST | `/api/cycle/save` | Save cycle + detect irregularity |
| GET  | `/api/cycle/history` | Last 6 cycles |
| GET  | `/api/cycle/current` | Current phase |
| POST | `/api/cycle/symptoms/log` | Log daily symptoms |
| GET  | `/api/cycle/symptoms/:userId` | Get symptom history |

### Pregnancy
| Method | Route | Description |
|--------|-------|-------------|
| POST | `/api/pregnancy/start` | Start pregnancy tracking |
| GET  | `/api/pregnancy/status` | Current week/trimester |
| GET  | `/api/pregnancy/checklist/:week` | Tasks up to given week |
| PUT  | `/api/pregnancy/checklist/update` | Mark task complete |
| POST | `/api/pregnancy/contraction/start` | Start timer |
| POST | `/api/pregnancy/contraction/stop` | Stop + record |
| GET  | `/api/pregnancy/contraction/history` | All contractions |

### Vaccines
| Method | Route | Description |
|--------|-------|-------------|
| GET  | `/api/vaccines` | All vaccines for user |
| POST | `/api/vaccines/add` | Add a vaccine |
| PUT  | `/api/vaccines/update-status` | Mark complete + generate QR |
| POST | `/api/vaccines/auto-schedule` | Auto-generate from LMP |
| GET  | `/api/vaccines/verify/:id` | Public QR verification |

### Nutrition
| Method | Route | Description |
|--------|-------|-------------|
| POST | `/api/nutrition/plan` | Get personalised meal plan |

### Community
| Method | Route | Description |
|--------|-------|-------------|
| POST | `/api/community/post` | Create a post |
| GET  | `/api/community/posts` | Feed (with ?category=) |
| POST | `/api/community/like` | Toggle like |
| POST | `/api/community/comment` | Add comment |

### Risk Assessment
| Method | Route | Description |
|--------|-------|-------------|
| POST | `/api/risk/predict` | ML/rule-based risk prediction |
| GET  | `/api/risk/history` | Past assessments |

### AI Assistant
| Method | Route | Description |
|--------|-------|-------------|
| POST | `/api/assistant/chat` | Send message + get response |
| GET  | `/api/assistant/alerts` | Proactive health alerts |

### ML Service (FastAPI on :8000)
| Method | Route | Description |
|--------|-------|-------------|
| POST | `/predict-risk` | ML risk prediction |
| GET  | `/health` | Service health check |

---

## 🧪 Sample API Requests & Responses

### 1. Vaccine Creation
**Request:**
```json
POST /api/vaccines/add
Authorization: Bearer <token>

{
  "vaccineName": "Tdap (Whooping Cough)",
  "scheduledDate": "2024-06-15",
  "weekNumber": 28,
  "trimester": 3,
  "notes": "Required between weeks 27-36"
}
```
**Response:**
```json
{
  "success": true,
  "vaccine": {
    "_id": "665a1b2c3d4e5f6a7b8c9d0e",
    "userId": "665a1b2c...",
    "vaccineName": "Tdap (Whooping Cough)",
    "scheduledDate": "2024-06-15T00:00:00.000Z",
    "status": "pending",
    "weekNumber": 28,
    "trimester": 3
  }
}
```

---

### 2. Nutrition Plan
**Request:**
```json
POST /api/nutrition/plan
Authorization: Bearer <token>

{
  "weight": 65,
  "height": 162,
  "trimester": 2,
  "dietType": "vegetarian"
}
```
**Response:**
```json
{
  "success": true,
  "plan": {
    "bmi": 24.8,
    "calories": 2340,
    "dailyPlan": {
      "breakfast": "Poha with peas & peanuts",
      "lunch": "Paneer sabzi + roti + salad",
      "dinner": "Palak paneer + roti + dal",
      "snacks": "Roasted chana"
    },
    "nutrients": {
      "protein": "60-75g",
      "calcium": "1000mg",
      "iron": "27mg",
      "folicAcid": "600mcg",
      "omega3": "200mg DHA"
    },
    "groceryList": ["Whole grains", "Lentils", "Leafy vegetables", "Paneer", "Peanuts", ...],
    "hydration": "8-10 glasses of water/day",
    "trimesterTips": "Increase iron and calcium. Baby's growth accelerates."
  }
}
```

---

### 3. Community Post
**Request:**
```json
POST /api/community/post
Authorization: Bearer <token>

{
  "content": "Anyone else experiencing back pain in their second trimester? Looking for gentle yoga tips! 🤰",
  "category": "pregnancy",
  "isAnonymous": false,
  "tags": ["backpain", "yoga", "pregnancy"]
}
```
**Response:**
```json
{
  "success": true,
  "post": {
    "_id": "665a1b2c3d4e5f6a7b8c9d0f",
    "userId": { "_id": "...", "name": "Priya Sharma", "isVerifiedExpert": false },
    "content": "Anyone else experiencing back pain in their second trimester?...",
    "category": "pregnancy",
    "likes": [],
    "comments": [],
    "isAnonymous": false,
    "createdAt": "2024-06-10T09:30:00.000Z"
  }
}
```

---

### 4. Risk Assessment
**Request:**
```json
POST /api/risk/predict
Authorization: Bearer <token>

{
  "age": 32,
  "systolicBP": 148,
  "diastolicBP": 94,
  "bloodSugar": 155,
  "bodyTemp": 98.6,
  "heartRate": 88
}
```
**Response:**
```json
{
  "success": true,
  "result": {
    "riskLevel": "High",
    "confidence": 0.91,
    "detectedCondition": "Preeclampsia risk (High Blood Pressure), Gestational Diabetes risk",
    "recommendations": [
      "🚨 Seek immediate medical attention",
      "📞 Call your OB/GYN now",
      "🏥 Consider emergency visit",
      "🧘 Practice stress-reduction techniques",
      "🥗 Follow gestational diabetes diet"
    ]
  }
}
```

---

## 🎨 Design System

| Token | Value |
|-------|-------|
| Primary Pink | `#ec4899` |
| Lavender | `#a855f7` |
| Glass BG | `rgba(255,255,255,0.6)` |
| Glass Border | `rgba(255,255,255,0.8)` |
| Blur | `backdrop-filter: blur(20px)` |
| Font | Inter + Poppins |

---

## 🤖 Face Verification Notes

The `FaceVerification` component in `frontend/src/components/FaceVerification.jsx`:

1. **Attempts to load** `face-api.js` models from the official CDN
2. **If CDN succeeds** → real-time gender detection via `TinyFaceDetector` + `AgeGenderNet`
3. **If CDN fails** → falls back to simulation mode (female assumed for demo)
4. **For production** → download models to `public/models/` and update the `MODEL_URL` constant

To use local models:
```bash
# Download models to frontend/public/models/
# Update FaceVerification.jsx line:
const MODEL_URL = "/models";
```

---

## 📜 License

MIT — Built with 🌸 for women's health
