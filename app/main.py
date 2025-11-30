from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import RedirectResponse
import httpx
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Allow CORS for development convenience
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

class GenerateRequest(BaseModel):
    prompt: str
    endpoint_url: str
    max_new_tokens: int = 200

def build_endpoint(endpoint_url: str, target_path: str) -> str:
    """Ensure requests hit the desired endpoint even if user omits the path."""
    cleaned = endpoint_url.strip()
    if not cleaned:
        return cleaned
    base = cleaned[:-1] if cleaned.endswith("/") else cleaned
    # If user already provided /generate or /generate_alt, strip it so we can re-append the target path.
    for suffix in ("/generate_alt", "/generate"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    if base.endswith(target_path):
        return base
    return f"{base}{target_path}"


async def call_colab_endpoint(endpoint_url: str, prompt: str, max_new_tokens: int):
    """
    Sends a request to the Colab ngrok endpoint.
    """
    prompt_body = prompt.strip()
    messages = [{"role": "user", "content": prompt_body}]
    generation_params = {
        "max_new_tokens": max_new_tokens,
        "temperature": 0.1,
        "top_p": 0.9,
    }
    
    payload = {
        "messages": messages,
        "prompt": prompt_body,  # fallback for simpler APIs
        **generation_params,
    }

    logger.info(f"Sending request to {endpoint_url}")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(endpoint_url, json=payload)
            response.raise_for_status()
            data = response.json()
            
            # Attempt to parse response from common formats
            if "choices" in data and len(data["choices"]) > 0:
                # OpenAI format
                content = data["choices"][0].get("message", {}).get("content", "")
                if not content:
                    content = data["choices"][0].get("text", "")
                return content
            elif "generated_text" in data:
                # HuggingFace Inference API format
                return data["generated_text"]
            elif "response" in data:
                # Simple custom format
                return data["response"]
            else:
                # Return raw JSON if unknown format
                return str(data)
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.text}")
            return f"Error: {e.response.status_code} - {e.response.text}"
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
            return f"Connection Error: {str(e)}"

@app.post("/api/generate")
async def generate(request: GenerateRequest):
    # We launch both requests in parallel for better performance
    import asyncio
    
    primary_endpoint = build_endpoint(request.endpoint_url, "/generate")
    secondary_endpoint = build_endpoint(request.endpoint_url, "/generate_alt")

    primary_task = call_colab_endpoint(primary_endpoint, request.prompt, request.max_new_tokens)
    secondary_task = call_colab_endpoint(secondary_endpoint, request.prompt, request.max_new_tokens)
    
    primary_res, secondary_res = await asyncio.gather(primary_task, secondary_task)
    
    return {
        "primary": primary_res,
        "secondary": secondary_res
    }

@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
