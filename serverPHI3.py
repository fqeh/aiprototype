from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# Load Microsoft Phi-3-mini model
print("Loading Microsoft Phi-3-mini...")
phi_tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
phi_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct", 
    torch_dtype=torch.float16, 
    device_map="auto",
    trust_remote_code=True
)
phi_pipeline = pipeline("text-generation", model=phi_model, tokenizer=phi_tokenizer)

# Available models
models = {
    "phi-3-mini": phi_pipeline
}

# Request schema
class PromptRequest(BaseModel):
    model: str
    prompt: str
    max_new_tokens: int = 200

def generate_text(model_name: str, prompt: str, max_new_tokens: int = 200):
    if model_name not in models:
        return f"Error: Model {model_name} not available"
    
    pipeline = models[model_name]
    result = pipeline(
        prompt, 
        max_new_tokens=max_new_tokens, 
        do_sample=True, 
        temperature=0.1,
        pad_token_id=pipeline.tokenizer.eos_token_id
    )
    return result[0]['generated_text']

if __name__ == "__main__":
    # Test the model
    test_prompt = input("Write a prompt here: ")
    
    print("Testing Microsoft Phi-3-mini...")
    result = generate_text("phi-3-mini", test_prompt)
    print(f"Result: {result}")