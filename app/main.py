from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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

async def call_colab_endpoint(endpoint_url: str, prompt: str, style: str):
    """
    Sends a request to the Colab ngrok endpoint.
    Constructs the prompt with the appropriate style tag and message format used by eval_style_responses.
    """
    styled_prompt = f"<style:{style}> {prompt}".strip()
    messages = [{"role": "user", "content": styled_prompt}]
    generation_params = {
        "max_new_tokens": 200,
        "temperature": 0.7,
        "top_p": 0.9,
    }
    
    payload = {
        "messages": messages,
        "prompt": styled_prompt,  # fallback for simpler APIs
        **generation_params,
    }

    logger.info(f"Sending request to {endpoint_url} with style {style}")
    
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
    
    standard_task = call_colab_endpoint(request.endpoint_url, request.prompt, "none")
    chosun_task = call_colab_endpoint(request.endpoint_url, request.prompt, "chosun")
    
    standard_res, chosun_res = await asyncio.gather(standard_task, chosun_task)
    
    return {
        "standard": standard_res,
        "chosun": chosun_res
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
