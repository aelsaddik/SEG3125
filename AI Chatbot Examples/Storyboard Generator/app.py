import gradio as gr
from groq import Groq
import os

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are an expert storyboard generator. Create detailed storyboards with:
- Shot numbers, visual descriptions, camera angles
- Actions, dialogue, sound effects, timing
Format as markdown tables. Be creative and cinematic."""

def respond(message, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        if h[1]:
            messages.append({"role": "assistant", "content": h[1]})
    
    messages.append({"role": "user", "content": message})
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.9,
            max_completion_tokens=2048,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# Simple ChatInterface - most compatible
demo = gr.ChatInterface(
    fn=respond,
    title="🎬 Storyboard Generator AI",
    description="Create professional storyboards for films, animations, and more!",
    examples=[
        "Create a storyboard for a 30-second coffee commercial",
        "Generate a horror movie opening scene storyboard",
        "Design a storyboard for a romantic comedy meet-cute at a bookstore",
    ],
    theme="soft",
)

if __name__ == "__main__":
    demo.launch()
