import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Initialize the FastAPI app
app = FastAPI()

# Allow CORS for all origins (you can restrict this to specific origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to a list of specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure the Generative AI model
genai.configure(api_key=os.getenv("API_KEY"))

generation_config = {
    "temperature": 0,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
    system_instruction="ai name is aurora and Aurora should respond in a friendly, concise, and informative manner."
)

# Initialize the chat session
history = []
chat_session = model.start_chat(history=history)

# Define the request body model
class ChatRequest(BaseModel):
    message: str

# Define the /chat endpoint
@app.post("/chat")
async def chat(request: ChatRequest):
    user_input = request.message

    if user_input.lower() in ['exit', 'quit', 'bye']:
        return {"response": "Goodbye!"}

    try:
        response = chat_session.send_message(user_input)
        model_response = response.text

        history.append({"role": "user", "parts": [user_input]})
        history.append({"role": "model", "parts": [model_response]})

        return {"response": model_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
