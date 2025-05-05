from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import os

class ChatRequest(BaseModel):
    message: str

app = FastAPI()

# URL of the defense proxy
DEFENSE_URL = os.getenv("DEFENSE_URL", "http://defense:9000/proxy")

@app.post("/chat")
async def chat(chat_request: ChatRequest):
    # Forward message through defense proxy to model
    async with httpx.AsyncClient() as client:
        resp = await client.post(DEFENSE_URL, json={"message": chat_request.message})
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()
