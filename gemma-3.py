from transformers import pipeline
import torch
import os
import sys
import threading
import queue
import time
from datetime import datetime
from flask import Flask, request, jsonify
from concurrent.futures import ThreadPoolExecutor
import logging

# --- do not change: torch setup ---
device = torch.device("cuda")
x = torch.rand(10, 10).to(device)
y = torch.rand(10, 10).to(device)
z = x @ y
# --- end: torch setup ---

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model dictionary - Qwen (text) + Gemma (vision+text)
MODELS = {
    "1": {
        "name": "Qwen 2.5 1.5B Instruct",
        "id": "Qwen/Qwen2.5-1.5B-Instruct",
        "description": "Qwen's efficient 1.5B model",
        "size": "~3GB",
        "task": "text-generation",
    },
    "2": {
        "name": "Qwen 2.5 0.5B Instruct",
        "id": "Qwen/Qwen2.5-0.5B-Instruct",
        "description": "Qwen's smallest model, fast and lightweight",
        "size": "~1GB",
        "task": "text-generation",
    },
    "3": {
        "name": "Gemma-3 4B Instruct (VLM)",
        "id": "google/gemma-3-4b-it",
        "description": "Google Gemma 3, 4B instruct (vision + text)",
        "size": "~8–10GB",
        "task": "image-text-to-text",
    }
}

class ModelManager:
    """Manages multiple models and request queues"""

    def __init__(self, max_workers=2):
        self.models = {}  # {model_id: pipeline}
        self.queues = {}  # {model_id: queue.Queue()}
        self.locks = {}   # {model_id: threading.Lock()}
        self.processing_counts = {}  # {model_id: int}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")

    def _task_for_model(self, model_id: str) -> str:
        task = "text-generation"
        for _, v in MODELS.items():
            if v["id"] == model_id:
                task = v.get("task", "text-generation")
                break
        return task

    def load_model(self, model_id):
        """Load a model if not already loaded"""
        if model_id not in self.models:
            logger.info(f"Loading model: {model_id}")
            try:
                task = self._task_for_model(model_id)

                if task == "image-text-to-text":
                    # Gemma-3 (multimodal)
                    self.models[model_id] = pipeline(
                        "image-text-to-text",
                        model=model_id,
                        device="cuda",                 # or omit for CPU
                        dtype=torch.bfloat16,    
                        token=self.hf_token,
                        trust_remote_code=True,
                    )
                else:
                    # Qwen (text LLM)
                    self.models[model_id] = pipeline(
                        "text-generation",
                        model=model_id,
                        torch_dtype="auto",
                        device_map="auto",
                        token=self.hf_token,
                        trust_remote_code=True,
                    )

                self.queues[model_id] = queue.Queue()
                self.locks[model_id] = threading.Lock()
                self.processing_counts[model_id] = 0
                logger.info(f"Model {model_id} loaded successfully")

                # Start worker thread for this model
                threading.Thread(target=self._process_queue, args=(model_id,), daemon=True).start()

            except Exception as e:
                logger.error(f"Failed to load model {model_id}: {e}")
                raise

    def _process_queue(self, model_id):
        """Worker thread to process requests for a specific model"""
        model_task = self._task_for_model(model_id)

        while True:
            try:
                if model_id in self.queues:
                    task_item = self.queues[model_id].get(timeout=1)
                    if task_item is None:
                        break

                    payload, max_tokens, callback, request_id = task_item

                    with self.locks[model_id]:
                        self.processing_counts[model_id] += 1

                    logger.info(f"Processing request {request_id} for {model_id}")

                    try:
                        if model_task == "image-text-to-text":
                            # payload is {"messages": [...]}
                            result = self.models[model_id](
                                text=payload["messages"],
                                max_new_tokens=max_tokens,
                            )
                            # Last assistant message content
                            resp = result[0]["generated_text"][-1]["content"]
                            callback({"success": True, "response": resp, "model": model_id})
                        else:
                            # payload is a plain prompt string
                            prompt = payload
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

    def add_request(self, model_id, payload, max_tokens, callback, request_id):
        """Add a request to the queue"""
        if model_id not in self.models:
            self.load_model(model_id)

        queue_size = self.queues[model_id].qsize()
        if queue_size > 0:
            logger.info(f"Request {request_id} queued. Position: {queue_size + 1}")

        self.queues[model_id].put((payload, max_tokens, callback, request_id))
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

def select_models():
    """Display model menu and allow loading multiple models"""
    print("\n" + "="*60)
    print("="*60)

    for key, model in MODELS.items():
        print(f"\n[{key}] {model['name']}")
        print(f"    Size: {model['size']}")
        print(f"    {model['description']}")

    while True:
        choice = input("\nSelect option (1/2/3/q): ").strip()

        if choice.lower() == 'q':
            print("Exiting...")
            sys.exit(0)

        if choice == "1":
            return [MODELS["1"]["id"]]
        elif choice == "2":
            return [MODELS["2"]["id"]]
        elif choice == "3":
            return [MODELS["3"]["id"]]
        else:
            print("[ERROR] Invalid choice. Please select 1, 2, 3 or q")

# Initialize model manager
model_manager = ModelManager(max_workers=2)

# Select and load models
selected_models = select_models()

print(f"\n[INFO] Loading selected models...")
for model_id in selected_models:
    print(f"[INFO] Loading: {model_id}")
    try:
        model_manager.load_model(model_id)
        print(f"[OK] {model_id} loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load {model_id}: {e}")
        sys.exit(1)

# Flask app
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# Store request counter
request_counter = 0
request_lock = threading.Lock()

@app.post("/get-response/")
def get_response():
    global request_counter

    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt", "")
    max_new_tokens = int(data.get("max_new_tokens", 256))
    model_key = data.get("model_key", "")
    messages = data.get("messages")             # optional chat format for Gemma
    image_urls = data.get("image_urls", [])     # optional list of image URLs

    # Determine model
    if model_key and model_key in MODELS:
        model_id = MODELS[model_key]["id"]
        model_task = MODELS[model_key].get("task", "text-generation")
    else:
        model_id = data.get("model", None)
        if not model_id:
            model_id = list(model_manager.models.keys())[0] if model_manager.models else None
        # find task for arbitrary model_id
        model_task = "text-generation"
        for _, v in MODELS.items():
            if v["id"] == model_id:
                model_task = v.get("task", "text-generation")
                break

    if not model_id:
        return jsonify({"error": "No model specified or loaded"}), 400
    if model_id not in model_manager.models:
        return jsonify({"error": f"Model {model_id} not loaded. Available: {list(model_manager.models.keys())}"}), 400

    # Build payload by task
    if model_task == "image-text-to-text":
        if messages is None:
            # Build minimal chat from prompt + image URLs
            content = [{"type": "image", "url": u} for u in image_urls]
            if prompt:
                content.append({"type": "text", "text": prompt})
            if not content:
                return jsonify({"error": "For Gemma-3, provide 'messages' or 'prompt' with 'image_urls'."}), 400
            messages = [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user", "content": content},
            ]
        payload = {"messages": messages}
    else:
        if not prompt:
            return jsonify({"error": "Field 'prompt' is required for text-generation models."}), 400
        payload = prompt

    # Generate unique request ID
    with request_lock:
        request_counter += 1
        request_id = f"req_{request_counter}_{datetime.now().strftime('%H%M%S')}"

    # Create response container
    response_container = {"ready": False, "data": None}

    def callback(result):
        response_container["data"] = result
        response_container["ready"] = True

    # Add request to queue
    queue_position = model_manager.add_request(model_id, payload, max_new_tokens, callback, request_id)

    # Optional: log queue position
    if queue_position > 1:
        logger.info(f"Request queued: {request_id} (Position: {queue_position})")

    # Wait for response (with timeout)
    timeout = 300  # 5 minutes
    start_time = time.time()

    while not response_container["ready"]:
        if time.time() - start_time > timeout:
            return jsonify({"error": "Request timeout", "request_id": request_id}), 504
        time.sleep(0.1)

    result = response_container["data"]

    if result["success"]:
        return jsonify({
            "response": result["response"],
            "model": model_id,
            "request_id": request_id,
            "wait_time": round(time.time() - start_time, 2),
            "queue_position": queue_position
        })
    else:
        return jsonify({"error": result["error"], "request_id": request_id}), 500

@app.get("/status/")
def get_status():
    """Get server and queue status"""
    loaded_models = list(model_manager.models.keys())
    queue_status = model_manager.get_queue_status()

    return jsonify({
        "loaded_models": loaded_models,
        "queue_status": queue_status,
        "available_models": {k: v["name"] for k, v in MODELS.items()}
    })

@app.get("/")
def home():
    """Simple status page"""
    loaded = list(model_manager.models.keys())
    models_html = ''.join([f"<li>[{k}] {v['name']} ({v['size']}) - {'LOADED' if v['id'] in loaded else 'NOT LOADED'}</li>" for k, v in MODELS.items()])

    return f"""
    <h1>Qwen + Gemma Multi-Model Server</h1>
    <p>Loaded Models: {', '.join(loaded) if loaded else 'None'}</p>

    <h2>Available Endpoints:</h2>
    <ul>
        <li>POST /get-response/ - Generate text (1/2) or vision+text (3)</li>
        <li>GET /status/ - Check server status</li>
    </ul>

    <h2>Models:</h2>
    <ul>
        {models_html}
    </ul>
    <h3>Examples:</h3>
    <pre>
    # Qwen (text)
    curl -X POST http://0.0.0.0:5009/get-response/ \\
      -H "Content-Type: application/json" \\
      -d '{{"prompt": "Hello", "model_key": "1", "max_new_tokens": 50}}'

    # Gemma-3 (vision + text, via image URL)
    curl -X POST http://0.0.0.0:5009/get-response/ \\
      -H "Content-Type: application/json" \\
      -d '{{"model_key":"3","prompt":"What animal is on the candy?","image_urls":["https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"],"max_new_tokens":80}}'
    </pre>
    """

if __name__ == "__main__":
    import os

    # Read host/port from environment (default to 0.0.0.0:5009)
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "5009"))

    print("\n" + "="*60)
    print("SERVER READY")
    print("="*60)
    print(f"Address: http://{HOST}:{PORT}")
    print(f"Loaded Models: {list(model_manager.models.keys())}")
    print("Multithreading: Enabled")
    print("Queue System: Active per model")
    print("\nAPI Usage:")
    print('  Use model_key="1" for Qwen 1.5B')
    print('  Use model_key="2" for Qwen 0.5B')
    print('  Use model_key="3" for Gemma-3 4B (vision+text)')
    print("="*60 + "\n")

    app.run(host=HOST, port=PORT, threaded=True)
