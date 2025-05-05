from fastapi import FastAPI, Request, HTTPException
import httpx

app = FastAPI()

# Guardrail proxy: inspect requests and forward to LLM model
MODEL_URL = "http://model:8000/chat"

@app.post("/proxy")
async def proxy(request: Request):
    body = await request.json()
    message = body.get("message")
    if not message:
        raise HTTPException(status_code=400, detail="Missing 'message' field")
    # Example guardrail: reject messages that contain banned words
    banned = ["hack", "exploit"]
    if any(word in message.lower() for word in banned):
        raise HTTPException(status_code=403, detail="Forbidden content detected")
    # Forward the request to the model
    async with httpx.AsyncClient() as client:
        resp = await client.post(MODEL_URL, json={"message": message})
    return resp.json()
