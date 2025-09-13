# server.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

# Load models once at startup
print("Loading gpt-oss...")
gpt_tokenizer = AutoTokenizer.from_pretrained("path/to/gpt-oss")
gpt_model = AutoModelForCausalLM.from_pretrained("path/to/gpt-oss", torch_dtype=torch.float16, device_map="auto")
gpt_pipeline = pipeline("text-generation", model=gpt_model, tokenizer=gpt_tokenizer)

print("Loading lama...")
lama_tokenizer = AutoTokenizer.from_pretrained("path/to/lama")
lama_model = AutoModelForCausalLM.from_pretrained("path/to/lama", torch_dtype=torch.float16, device_map="auto")
lama_pipeline = pipeline("text-generation", model=lama_model, tokenizer=lama_tokenizer)

# Available models
models = {
    "gpt-oss": gpt_pipeline,
    "lama": lama_pipeline
}

# Request schema
class PromptRequest(BaseModel):
    model: str
    prompt: str
    max_new_tokens: int = 200

@app.post("/generate")
async def generate(req: PromptRequest):
    if req.model not in models:
        return {"error": f"Model {req.model} not available. Choose from {list(models.keys())}"}

    generator = models[req.model]
    output = generator(
        req.prompt,
        max_new_tokens=req.max_new_tokens,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )[0]["generated_text"]

    return {"model": req.model, "prompt": req.prompt, "response": output}
