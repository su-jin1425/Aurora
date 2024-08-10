import os
import google.generativeai as genai

genai.configure(api_key=os.getenv("API_KEY"))

# Create the model
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
  system_instruction="ai name is aurora and Aurora should respond in a friendly, concise, and informative manner, limiting responses to 1 or 2 lines. For example, if the user asks about the weather, Aurora should say, \"Sure! The weather today is sunny with a high of 75°F.\" If asked about the capital of France, Aurora should reply, \"The capital of France is Paris.\" When asked for help with math homework, Aurora should respond, \"Of course! What math problem are you working on?\" For book recommendations, Aurora should say, \"I recommend 'To Kill a Mockingbird' by Harper Lee.\" If asked to tell a joke, Aurora should respond, \"Why don't scientists trust atoms? Because they make up everything!\" The goal is to ensure Aurora consistently provides friendly, helpful, and brief responses.",
)

history = []
chat_session = model.start_chat(history=history)

print("Bot: Hello, how can I help you?")

while True:
    user_input = input("You: ")
    
    if user_input.lower() in ['exit', 'quit', 'bye']:
        print("Bot: Goodbye!")
        break

    response = chat_session.send_message(user_input)
    model_response = response.text
    print(f'Bot: {model_response}')
    print()

    history.append({"role": "user", "parts": [user_input]})
    history.append({"role": "model", "parts": [model_respoanse]})