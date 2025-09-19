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

# Model dictionary
MODELS = {
    "1": {
        "name": "Qwen 2.5 1.5B Instruct",
        "id": "Qwen/Qwen2.5-1.5B-Instruct",
        "description": "Qwen's efficient 1.5B model",
        "size": "~3GB"
    },
    "2": {
        "name": "Qwen 2.5 0.5B Instruct",
        "id": "Qwen/Qwen2.5-0.5B-Instruct",
        "description": "Qwen's smallest model, fast and lightweight",
        "size": "~1GB"
    },
    "3": {
        "name": "Gemma 3 4B IT",
        "id": "google/gemma-3-4b-it",
        "description": "Google's efficient 2B instruction-tuned model",
        "size": "~4GB"
    },
    "4": {
        "name": "GPT OSS 20B",
        "id": "openai/gpt-oss-20b",
        "description": "OpenAI's open-source 20B parameter model",
        "size": "~40GB"
    },
    "5": {
        "name": "Meta Llama 3.1 8B Instruct",
        "id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "description": "Meta's instruction-tuned 8B parameter model",
        "size": "~16GB"
    }
}

def process_prompt_format(data):
    """
    Processes different prompt formats and converts them to simple text
    """
    prompt_data = data.get("prompt", "")

    # Simple format: "prompt": "hello"
    if isinstance(prompt_data, str):
        return prompt_data.strip()

    # Complex format: "prompt": [{"role": "user", "prompts": "hello"}]
    elif isinstance(prompt_data, list):
        if len(prompt_data) == 0:
            return ""

        # Process list of messages
        processed_parts = []
        for item in prompt_data:
            if isinstance(item, dict):
                # Extract prompt content
                content = item.get("prompts", "") or item.get("content", "") or item.get("message", "")
                role = item.get("role", "")

                if content:
                    if role:
                        processed_parts.append(f"{role}: {content}")
                    else:
                        processed_parts.append(content)

        return "\n".join(processed_parts).strip()

    # Object format: "prompt": {"content": "hello"}
    elif isinstance(prompt_data, dict):
        content = prompt_data.get("prompts", "") or prompt_data.get("content", "") or prompt_data.get("message", "")
        return content.strip()

    return ""

def map_external_model_to_internal(external_model):
    """
    Maps external models to internal available ones
    """
    model_mappings = {
        "meta-llama/Llama-3.1-8B": "5",  # Map to new Meta Llama model
        "meta-llama/Meta-Llama-3.1-8B-Instruct": "5",  # Direct mapping
        "llama-3.1": "5",
        "llama-3.1-8b": "5",
        "llama": "5",
        "qwen-1.5b": "1",
        "qwen-0.5b": "2",
        "small": "2",
        "large": "5"  # Changed from "1" to "5" for larger model
    }

    # Search for exact mapping
    if external_model in model_mappings:
        return model_mappings[external_model]

    # Search for partial mapping
    external_lower = external_model.lower()
    for external_key, internal_key in model_mappings.items():
        if external_key.lower() in external_lower:
            return internal_key

    # Default to model 5 (Meta Llama 3.1 8B) instead of model 1
    return "5"

class ModelManager:
    """Manages multiple models and request queues"""

    def __init__(self, max_workers=2):
        self.models = {}  # {model_id: pipeline}
        self.queues = {}  # {model_id: queue.Queue()}
        self.locks = {}   # {model_id: threading.Lock()}
        self.processing_counts = {}  # {model_id: int}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.hf_token = ""

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
                    token=self.hf_token
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
        while True:
            try:
                if model_id in self.queues:
                    task = self.queues[model_id].get(timeout=1)
                    if task is None:
                        break

                    prompt, max_tokens, callback, request_id = task

                    with self.locks[model_id]:
                        self.processing_counts[model_id] += 1
                        count = self.processing_counts[model_id]

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

def select_models():
    """Display model menu and allow loading multiple models"""
    print("\n" + "="*60)
    print("="*60)

    for key, model in MODELS.items():
        print(f"\n[{key}] {model['name']}")
        print(f"    Size: {model['size']}")
        print(f"    {model['description']}")

    while True:
        choice = input("\nSelect option (1/2/3/4/5/q): ").strip()

        if choice.lower() == 'q':
            print("Exiting...")
            sys.exit(0)

        if choice == "1":
            return [MODELS["1"]["id"]]
        elif choice == "2":
            return [MODELS["2"]["id"]]
        elif choice == "3":
            return [MODELS["3"]["id"]]
        elif choice == "4":
            return [MODELS["4"]["id"]]
        elif choice == "5":
            return [MODELS["5"]["id"]]
        else:
            print("[ERROR] Invalid choice. Please select 1, 2, 3, 4, 5 or q")

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

@app.route("/get-response/", methods=["POST", "GET"])
@app.route("/get-response", methods=["POST", "GET"])
def get_response():
    global request_counter

    # Handle GET requests
    if request.method == "GET":
        return jsonify({
            "error": "Method not allowed",
            "message": "This endpoint requires POST method",
            "supported_formats": [
                {
                    "simple": {
                        "prompt": "Hello world",
                        "model_key": "5",
                        "max_new_tokens": 100
                    }
                },
                {
                    "complex": {
                        "prompt": [{"role": "user", "prompts": "Hello world"}],
                        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                        "max_new_tokens": 100
                    }
                }
            ]
        }), 405

    # Validate Content-Type
    if not request.is_json:
        return jsonify({
            "error": "Invalid Content-Type",
            "message": "Content-Type must be application/json",
            "received": request.content_type or "None"
        }), 400

    # Parse JSON
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Empty JSON body"}), 400
    except Exception as e:
        return jsonify({
            "error": "Invalid JSON",
            "message": str(e)
        }), 400

    # Process prompt (handles both simple and complex formats)
    try:
        prompt = process_prompt_format(data)
        if not prompt:
            return jsonify({
                "error": "Missing or invalid prompt",
                "message": "Prompt is required. Supported formats: string, array of objects with 'prompts' field",
                "received": data.get("prompt")
            }), 400
    except Exception as e:
        return jsonify({
            "error": "Prompt processing error",
            "message": str(e),
            "received": data.get("prompt")
        }), 400

    # Handle max_new_tokens
    max_new_tokens = data.get("max_new_tokens", data.get("max_tokens", 256))
    try:
        max_new_tokens = int(max_new_tokens)
        if max_new_tokens <= 0 or max_new_tokens > 2048:
            raise ValueError("Must be between 1 and 2048")
    except (ValueError, TypeError):
        max_new_tokens = 256  # Default fallback

    # Determine model (handle both internal and external model references)
    model_key = None
    model_id = None

    # Check for internal model_key first
    if "model_key" in data:
        model_key = str(data["model_key"]).strip()
        if model_key in MODELS:
            model_id = MODELS[model_key]["id"]

    # Check for external model reference
    elif "model" in data:
        external_model = str(data["model"]).strip()
        model_key = map_external_model_to_internal(external_model)
        model_id = MODELS[model_key]["id"]
        logger.info(f"Mapped external model '{external_model}' to internal model_key '{model_key}' ({model_id})")

    # Fallback to first available model
    if not model_id and model_manager.models:
        model_id = list(model_manager.models.keys())[0]
        model_key = "5"  # Default to Meta Llama instead of "1"

    if not model_id:
        return jsonify({
            "error": "No model available",
            "available_models": {k: v["name"] for k, v in MODELS.items()},
            "message": "No models are currently loaded"
        }), 500

    if model_id not in model_manager.models:
        return jsonify({
            "error": "Model not loaded",
            "requested": model_id,
            "loaded_models": list(model_manager.models.keys())
        }), 400

    # Generate unique request ID
    with request_lock:
        request_counter += 1
        request_id = f"req_{request_counter}_{datetime.now().strftime('%H%M%S')}"

    logger.info(f"Processing request {request_id}: model={model_id}, prompt_length={len(prompt)}")

    # Create response container
    response_container = {"ready": False, "data": None}

    def callback(result):
        response_container["data"] = result
        response_container["ready"] = True

    try:
        queue_position = model_manager.add_request(model_id, prompt, max_new_tokens, callback, request_id)

        timeout = 300
        start_time = time.time()

        while not response_container["ready"]:
            if time.time() - start_time > timeout:
                return jsonify({
                    "error": "Request timeout",
                    "request_id": request_id
                }), 504
            time.sleep(0.1)

        result = response_container["data"]

        if result["success"]:
            wait_time = round(time.time() - start_time, 2)
            logger.info(f"Request {request_id} completed in {wait_time}s")

            return jsonify({
                "response": result["response"],
                "model": model_id,
                "model_name": MODELS.get(model_key, {}).get("name", model_id),
                "original_model_request": data.get("model", model_key),
                "request_id": request_id,
                "wait_time": wait_time,
                "queue_position": queue_position,
                "prompt_length": len(prompt),
                "response_length": len(result["response"])
            })
        else:
            return jsonify({
                "error": "Model processing failed",
                "details": result["error"],
                "request_id": request_id
            }), 500

    except Exception as e:
        logger.error(f"Unexpected error for request {request_id}: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "request_id": request_id
        }), 500

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
    <h1>Qwen Multi-Model Server</h1>
    <p>Loaded Models: {', '.join(loaded) if loaded else 'None'}</p>
    
    <h2>Available Endpoints:</h2>
    <ul>
        <li>POST /get-response/ - Generate text (specify model_key: "1", "2", "3", "4", or "5")</li>
        <li>GET /status/ - Check server status</li>
    </ul>
    
    <h2>Models:</h2>
    <ul>
        {models_html}
    </ul>
    
    <h3>Supported Formats:</h3>
    <h4>Simple Format:</h4>
    <pre>
    curl -X POST http://0.0.0.0:5001/get-response/ \\
      -H "Content-Type: application/json" \\
      -d '{{"prompt": "Hello", "model_key": "5", "max_new_tokens": 50}}'
    </pre>
    
    <h4>Complex Format:</h4>
    <pre>
    curl -X POST http://0.0.0.0:5001/get-response/ \\
      -H "Content-Type: application/json" \\
      -d '{{"prompt": [{{"role": "user", "prompts": "Hello"}}], "model": "meta-llama/Meta-Llama-3.1-8B-Instruct", "max_new_tokens": 50}}'
    </pre>
    """

if __name__ == "__main__":
    PORT = 5001

    print(f"\n" + "="*60)
    print(f"SERVER READY")
    print(f"="*60)
    print(f"Address: http://0.0.0.0:{PORT}")
    print(f"Loaded Models: {list(model_manager.models.keys())}")
    print(f"Multithreading: Enabled")
    print(f"Queue System: Active per model")
    print(f"\nAPI Usage:")
    print(f'  Use model_key="1" for Qwen 1.5B')
    print(f'  Use model_key="2" for Qwen 0.5B')
    print(f'  Use model_key="3" for gemma-3-4b-it')
    print(f'  Use model_key="4" for gpt-oss-20b')
    print(f'  Use model_key="5" for Meta-Llama-3.1-8B-Instruct')
    print(f"\nSupported prompt formats:")
    print(f'  Simple: {{"prompt": "Hello world"}}')
    print(f'  Complex: {{"prompt": [{{"role": "user", "prompts": "Hello world"}}]}}')
    print(f"="*60 + "\n")

    app.run(host="0.0.0.0", port=PORT, threaded=True)
