from transformers import pipeline
import torch
import threading
import queue
import time
from datetime import datetime
from flask import Flask, request, jsonify
from concurrent.futures import ThreadPoolExecutor
import logging
import requests
from huggingface_hub import login 

# --- Torch setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.rand(10, 10).to(device)
y = torch.rand(10, 10).to(device)
z = x @ y
# ----------------

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# ModelManager
class ModelManager:
    """Manages multiple models with queues and threading"""

    def __init__(self, max_workers=2, hf_token=None):
        self.models = {}             # {model_id: pipeline}
        self.queues = {}             # {model_id: queue.Queue()}
        self.locks = {}              # {model_id: threading.Lock()}
        self.processing_counts = {}  # {model_id: int}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.hf_token = hf_token

    def load_model(self, model_id):
        """Load a model if not already loaded"""
        if model_id not in self.models:
            logger.info(f"Loading model: {model_id}")
            try:
                self.models[model_id] = pipeline(
                    "text-generation",
                    model=model_id,
                    torch_dtype="auto",
                    device_map="auto",
                    use_auth_token=self.hf_token
                )
                self.queues[model_id] = queue.Queue()
                self.locks[model_id] = threading.Lock()
                self.processing_counts[model_id] = 0

                # Start worker thread for this model
                threading.Thread(target=self._process_queue, args=(model_id,), daemon=True).start()
                logger.info(f"Model {model_id} loaded successfully")

            except Exception as e:
                logger.error(f"Failed to load model {model_id}: {e}")
                raise

    def _process_queue(self, model_id):
        """Worker thread to process requests for a specific model"""
        while True:
            try:
                task = self.queues[model_id].get(timeout=1)
                if task is None:
                    break

                prompt, max_tokens, callback, request_id = task

                with self.locks[model_id]:
                    self.processing_counts[model_id] += 1

                logger.info(f"Processing request {request_id} for {model_id}")
                try:
                    result = self.models[model_id](
                        prompt,
                        max_new_tokens=max_tokens,
                    )
                    generated_text = result[0]["generated_text"][len(prompt):]
                    callback({"success": True, "response": generated_text, "model": model_id})
                except Exception as e:
                    callback({"success": False, "error": str(e)})
                finally:
                    with self.locks[model_id]:
                        self.processing_counts[model_id] -= 1

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Queue processing error: {e}")

    def add_request(self, model_id, prompt, max_tokens, callback, request_id):
        """Add a request to the queue"""
        if model_id not in self.models:
            self.load_model(model_id)
        queue_size = self.queues[model_id].qsize()
        if queue_size > 0:
            logger.info(f"Request {request_id} queued. Position: {queue_size + 1}")
        self.queues[model_id].put((prompt, max_tokens, callback, request_id))
        return queue_size + 1

    def get_queue_status(self):
        """Get current queue status for all models"""
        status = {}
        for model_id in self.models:
            status[model_id] = {
                "queue_size": self.queues[model_id].qsize(),
                "processing": self.processing_counts[model_id]
            }
        return status

# Initialize manager
model_manager = ModelManager(max_workers=2, hf_token="YOUR_HF_TOKEN")

# --- Prompt processing ---
def process_prompt_format(data):
    prompt_data = data.get("prompt", "")
    if isinstance(prompt_data, str):
        return prompt_data.strip()
    elif isinstance(prompt_data, list):
        processed_parts = []
        for item in prompt_data:
            if isinstance(item, dict):
                content = item.get("prompts") or item.get("content") or item.get("message") or ""
                role = item.get("role")
                if content:
                    processed_parts.append(f"{role}: {content}" if role else content)
        return "\n".join(processed_parts).strip()
    elif isinstance(prompt_data, dict):
        return (prompt_data.get("prompts") or prompt_data.get("content") or prompt_data.get("message") or "").strip()
    return ""

def map_external_model_to_internal(external_model, available_models):
    # Fallback: match exact or partial
    for m in available_models:
        if external_model.lower() in m["model_id"].lower():
            return m["model_id"]
    return available_models[0]["model_id"]  # default to first model

# --- Request counter ---
request_counter = 0
request_lock = threading.Lock()

# --- Flask endpoints ---
@app.route("/get-response/", methods=["POST", "GET"])
@app.route("/get-response", methods=["POST", "GET"])
def get_response():
    global request_counter

    if request.method == "GET":
        return jsonify({
            "error": "Method not allowed",
            "message": "This endpoint requires POST method"
        }), 405

    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json()
    prompt = process_prompt_format(data)
    if not prompt:
        return jsonify({"error": "Missing or invalid prompt"}), 400

    max_new_tokens = int(data.get("max_new_tokens", data.get("max_tokens", 256)))
    max_new_tokens = max(1, min(max_new_tokens, 2048))

    # Determine model
    model_id = data.get("model_key")
    if not model_id:
        model_id = data.get("model")  # external model name
        model_id = map_external_model_to_internal(model_id, available_models)

    if model_id not in model_manager.models:
        return jsonify({"error": "Model not loaded", "requested": model_id}), 400

    with request_lock:
        request_counter += 1
        request_id = f"req_{request_counter}_{datetime.now().strftime('%H%M%S')}"

    response_container = {"ready": False, "data": None}
    def callback(result):
        response_container["data"] = result
        response_container["ready"] = True

    queue_position = model_manager.add_request(model_id, prompt, max_new_tokens, callback, request_id)

    start_time = time.time()
    timeout = 300
    while not response_container["ready"]:
        if time.time() - start_time > timeout:
            return jsonify({"error": "Request timeout", "request_id": request_id}), 504
        time.sleep(0.1)

    result = response_container["data"]
    if result["success"]:
        wait_time = round(time.time() - start_time, 2)
        return jsonify({
            "response": result["response"],
            "model": model_id,
            "request_id": request_id,
            "wait_time": wait_time,
            "queue_position": queue_position,
            "prompt_length": len(prompt),
            "response_length": len(result["response"])
        })
    else:
        return jsonify({"error": "Model processing failed", "details": result["error"], "request_id": request_id}), 500

@app.get("/status/")
def get_status():
    return jsonify({
        "loaded_models": list(model_manager.models.keys()),
        "queue_status": model_manager.get_queue_status()
    })

@app.get("/")
def home():
    loaded = list(model_manager.models.keys())
    return f"<h1>Multi-Model Server</h1><p>Loaded Models: {', '.join(loaded) if loaded else 'None'}</p>"

# --- Main ---
def start_server():
    global available_models

    # Fetch models from API
    try:
        response = requests.get("https://serg.cs.uh.edu/aiprototype/api/models")
        response.raise_for_status()
        model_list = response.json()
        available_models = model_list.get("models", [])
        if not available_models:
            print("[ERROR] No models returned from API")
            exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to fetch models: {e}")
        exit(1)

    # Load models
    print(f"[INFO] Loading {len(available_models)} models from API...")
    for m in available_models:
        model_id = m.get("model_id")
        model_name = m.get("name", model_id)
        print(f"[INFO] Loading {model_name} ({model_id})")
        try:
            model_manager.load_model(model_id)
            print(f"[OK] {model_name} loaded successfully")
        except Exception as e:
            print(f"[ERROR] Failed to load {model_name}: {e}")

    # Start Flask server
    PORT = 5001
    print(f"\nSERVER READY at http://0.0.0.0:{PORT}")
    app.run(host="0.0.0.0", port=PORT, threaded=True)

if __name__ == "__main__":
    start_server()
