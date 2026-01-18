import gradio as gr
from groq import Groq
import os

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are an expert in storyboarding. Provide structured and insightful responses to queries 
about creating and refining storyboards"""

def respond(message, history, model, temperature, max_tokens):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        if h[1]:
            messages.append({"role": "assistant", "content": h[1]})
    
    messages.append({"role": "user", "content": message})
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# ChatInterface with additional inputs for parameters
demo = gr.ChatInterface(
    fn=respond,
    title="🎬 Storyboard Generator AI",
    description="Create professional storyboards for films, animations, and more!",
    additional_inputs=[
        gr.Dropdown(
            choices=[
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
            ],
            value="llama-3.3-70b-versatile",
            label="Model",
            info="Select the AI model to use"
        ),
        gr.Slider(
            minimum=0,
            maximum=2,
            value=0.9,
            step=0.1,
            label="Temperature",
            info="Controls randomness. Lower = more focused, Higher = more creative"
        ),
        gr.Slider(
            minimum=256,
            maximum=8192,
            value=2048,
            step=256,
            label="Max Tokens",
            info="Maximum length of the response"
        ),
    ],
    examples=[
        ["Create a storyboard for a 30-second coffee commercial"],
        ["Generate a horror movie opening scene storyboard"],
        ["Design a storyboard for a romantic comedy meet-cute at a bookstore"],
    ],
    theme="soft",
)

if __name__ == "__main__":
    demo.launch()
