# app.py
import os
import secrets
from datetime import datetime

from flask import Flask, request, jsonify, Response
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user,
    login_required, current_user
)
from pymongo import MongoClient
from bson import ObjectId
import requests
import bcrypt
from dotenv import load_dotenv

load_dotenv()

# ---------------- Config ----------------
MONGO_USER = os.getenv("MONGO_USER")
MONGO_PASS = os.getenv("MONGO_PASS")
DB_NAME = os.getenv("DB_NAME")
USERS_COLLECTION = os.getenv("USERS_COLLECTION")
API_KEYS_COLLECTION = os.getenv("API_KEYS_COLLECTION")
MODELS_COLLECTION = os.getenv("MODELS_COLLECTION")

LLM_SERVER_BASE = os.getenv("LLM_SERVER_BASE")
LLM_SERVER_ENDPOINT = os.getenv("LLM_SERVER_ENDPOINT")
CONNECT_TIMEOUT = float(os.getenv("CONNECT_TIMEOUT", "5"))
READ_TIMEOUT = float(os.getenv("READ_TIMEOUT", "60"))
PORT = int(os.getenv("PORT", "5000"))

# allow requests without API key for testing
ALLOW_NO_API_KEY_FOR_TESTS = os.getenv("ALLOW_NO_API_KEY_FOR_TESTS", "true").lower() == "true"

# Flask secrets (session cookies)
SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")

# ---------------- App + DB ----------------
app = Flask(__name__)
app.secret_key = SECRET_KEY

mongo_client = MongoClient(
    "mongodb://localhost:27090/Team309",
    username=MONGO_USER,
    password=MONGO_PASS,
    authSource="Team309",
    uuidRepresentation="standard",
)
db = mongo_client[DB_NAME]
users = db[USERS_COLLECTION]
api_keys = db[API_KEYS_COLLECTION]
models_col = db[MODELS_COLLECTION]

# helpful indexes (safe to run multiple times)
users.create_index("email", unique=True)
api_keys.create_index("user_id")
api_keys.create_index("key_id")
models_col.create_index("name", unique=True)
models_col.create_index("model_id", unique=True)

# ---------------- Flask-Login ----------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

class User(UserMixin):
    def __init__(self, _id, email):
        self.id = str(_id)      # Flask-Login expects a string id
        self.email = email

def _user_from_doc(doc):
    if not doc:
        return None
    return User(doc["_id"], doc.get("email"))

@login_manager.user_loader
def load_user(user_id: str):
    try:
        doc = users.find_one({"_id": ObjectId(user_id)})
    except Exception:
        return None
    return _user_from_doc(doc)

# ---------------- Helpers ----------------
def get_api_key_from_headers():
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth.split(" ", 1)[1].strip()
    return request.headers.get("X-API-Key", "").strip()

def find_api_key_doc(api_key_plain: str):
    """Return (doc_id, doc) if api key matches a stored hash; else (None, None)."""
    if not api_key_plain:
        return None, None
    key_b = api_key_plain.encode("utf-8")
    cursor = api_keys.find({}, projection={"api_key_hash": 1})
    for doc in cursor:
        h = doc.get("api_key_hash")
        if h and bcrypt.checkpw(key_b, h.encode("utf-8")):
            return doc["_id"], doc
    return None, None

def increment_api_key_usage(doc_id, model: str):
    if not doc_id or not model:
        return
    api_keys.update_one(
        {"_id": doc_id},
        {
            "$inc": {f"usage.{model}": 1},
            "$set": {"last_used_at": datetime.utcnow()}
        },
        upsert=False,
    )

def _serialize_user_doc(doc):
    return {
        "id": str(doc["_id"]),
        "email": doc.get("email")
    }

def _serialize_model_doc(doc):
    return {
        "id": str(doc["_id"]),
        "name": doc.get("name"),
        "model_id": doc.get("model_id"),
    }

# ---------------- Auth Endpoints ----------------
@app.post("/api/signup")
def signup():
    """
    Body: { "email": "...", "password": "..." }
    Creates a user, stores bcrypt password, generates an API key (returned once),
    and stores its hash in api_keys.
    """
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    if not email or not password:
        return jsonify({"error": "email and password are required"}), 400

    # prevent duplicate email
    if users.find_one({"email": email}):
        return jsonify({"error": "email already exists"}), 409

    pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode()
    user_doc = {
        "email": email,
        "password_hash": pw_hash,
        "created_at": datetime.utcnow(),
        "active": True
    }
    result = users.insert_one(user_doc)
    user_id = result.inserted_id

    # generate API key (return to client ONCE)
    api_key_plain = secrets.token_hex(32)
    api_key_hash = bcrypt.hashpw(api_key_plain.encode("utf-8"), bcrypt.gensalt()).decode()
    api_keys.insert_one({
        "user_id": user_id,
        "key_id": "primary",
        "api_key_hash": api_key_hash,
        "created_at": datetime.utcnow(),
        "usage": {}
    })

    # (optional) log user in immediately
    login_user(User(user_id, email))

    return jsonify({
        "message": "signup successful",
        "user": {"id": str(user_id), "email": email},
        "api_key": api_key_plain  # show only once now; never store plaintext
    }), 201

@app.post("/api/login")
def login():
    """
    Body: { "email": "...", "password": "..." }
    Validates credentials and starts a session cookie.
    """
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    if not email or not password:
        return jsonify({"error": "email and password are required"}), 400

    doc = users.find_one({"email": email, "active": True})
    if not doc:
        return jsonify({"error": "invalid credentials"}), 401

    pw_hash = doc.get("password_hash", "")
    if not pw_hash or not bcrypt.checkpw(password.encode("utf-8"), pw_hash.encode("utf-8")):
        return jsonify({"error": "invalid credentials"}), 401

    login_user(_user_from_doc(doc))
    return jsonify({"message": "login successful", "user": _serialize_user_doc(doc)}), 200

@app.post("/api/logout")
@login_required
def logout():
    logout_user()
    return jsonify({"message": "logged out"}), 200

@app.get("/api/me")
def me():
    if current_user.is_authenticated:
        return jsonify({"authenticated": True, "user": {"id": current_user.id, "email": getattr(current_user, "email", None)}})
    return jsonify({"authenticated": False})

# ---------------- Models Endpoint ----------------
@app.get("/api/models")
def list_models():
    """
    Returns all available models from the models collection.
    Response: { "models": [ { "id": "...", "name": "...", "model_id": "..." }, ... ] }
    """
    cursor = models_col.find({}, projection={"name": 1, "model_id": 1})
    items = [_serialize_model_doc(d) for d in cursor]
    return jsonify({"models": items}), 200

# ---------------- Existing Proxy (UNCHANGED) ----------------
@app.get("/api/healthz")
def healthz():
    return jsonify({"status": "ok"}), 200

@app.post("/api/get-response")
def get_response():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt")
    model = data.get("model")
    if not prompt or not model:
        return jsonify({"error": "Missing 'prompt' or 'model' in JSON body."}), 400

    # API key (optional for now)
    api_key = get_api_key_from_headers()
    key_doc_id, key_doc = (None, None)
    if api_key:
        key_doc_id, key_doc = find_api_key_doc(api_key)
        if not key_doc_id:
            return jsonify({"error": "Invalid API key"}), 401
        # per-model usage tracking
        increment_api_key_usage(key_doc_id, model)
    else:
        if not ALLOW_NO_API_KEY_FOR_TESTS:
            return jsonify({"error": "API key required"}), 401

    # forward to LLM server
    url = f"{LLM_SERVER_BASE}{LLM_SERVER_ENDPOINT}"
    try:
        upstream = requests.post(
            url,
            json={"prompt": prompt, "model": model},
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        )
    except requests.Timeout:
        return jsonify({"error": "Upstream timeout contacting LLM server."}), 504
    except requests.ConnectionError as e:
        return jsonify({"error": f"Connection error to LLM server: {str(e)}"}), 502

    # pass through response
    try:
        body = upstream.json()
        if isinstance(body, dict):
            body.setdefault("proxy_meta", {})
            body["proxy_meta"].update({
                "api_key_used": bool(api_key),
                "model": model,
                "status_from_llm": upstream.status_code
            })
        return jsonify(body), upstream.status_code
    except ValueError:
        return Response(
            upstream.text,
            status=upstream.status_code,
            content_type=upstream.headers.get("Content-Type", "text/plain")
        )

# ---------------- Run ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, threaded=True)
