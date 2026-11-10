import os
from dotenv import load_dotenv
from groq import Groq

# 1. Load environment variables
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file.")

# 2. Initialize Groq client
client = Groq(api_key=api_key)

# 3. Initialize conversation memory
# System prompt sets the personality and operational rules
messages = [
    {
        "role": "system",
        "content": "You are a sharp, conversational, and direct AI assistant. Keep responses clear and engaging."
    }
]

print("=" * 55)
print("Stage 1 Chatbot: Direct LLM with Multi-turn Memory")
print("Type 'exit' to quit | Type 'clear' to reset memory")
print("=" * 55 + "\n")

# 4. Interactive conversation loop
while True:
    user_input = input("You: ").strip()

    if not user_input:
        continue

    if user_input.lower() == "exit":
        print("\nExiting. Take care!")
        break

    if user_input.lower() == "clear":
        # Keep only the system prompt, clear chat history
        messages = [messages[0]]
        print("\n[Chat memory cleared]\n")
        continue

    # Add user message to history
    messages.append({"role": "user", "content": user_input})

    try:
        # Call Groq's high-speed inference engine
        response = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=messages,
            temperature=0.7
        )

        reply = response.choices[0].message.content

        # Append assistant reply to keep memory persistent
        messages.append({"role": "assistant", "content": reply})

        print(f"\nAI: {reply}\n")
        print("-" * 55)

    except Exception as e:
        print(f"\n[Error]: {e}\n")