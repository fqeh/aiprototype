from transformers import pipeline
import torch

# --- do not change: torch setup ---
device = torch.device("cuda")

x = torch.rand(10, 10).to(device)
y = torch.rand(10, 10).to(device)
z = x @ y
# --- end: torch setup ---

# Initialize the pipeline once at startup (left as-is)
model_id = "openai/gpt-oss-20b"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype="auto",
    device_map="cuda",
)

# --- Flask app replacing FastAPI ---
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.post("/get-response/")
def get_response():
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt", "")
    max_new_tokens = int(data.get("max_new_tokens", 256))

    if not prompt:
        return jsonify({"error": "Field 'prompt' is required."}), 400

    outputs = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
    )

    generated_text = outputs[0]["generated_text"][len(prompt):]
    return jsonify({"response": generated_text})

if __name__ == "__main__":
    # Start Flask server
    app.run(host="0.0.0.0", port=8000)
