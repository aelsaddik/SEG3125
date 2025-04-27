import gradio as gr
import os
from groq import Groq

# Initialize Groq client
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

# Initialize conversation history
conversation_history = []


def chat_with_bot_stream(user_input, language, response_style):
    global conversation_history

    if language == None:
        language = "English"

    if response_style == None:
        response_style = "Motivational"

    system_message = f"""
        You are a mentor and motivational chatbot. You help users with their goals by suggesting objectives and tasks.
    
        INSTRUCTIONS:
        - Keep individual chat responses relatively short, aim for under 400 characters unless necessary
        - If the user asks for help, provide clear, actionable steps
        - You respond to the user in the {language} language, and with a {response_style} style.
    """

    if conversation_history == []:
        conversation_history.insert(0, {"role": "system", "content": system_message})

    conversation_history.append({"role": "user", "content": user_input})

    completion = client.chat.completions.create(
        model="gemma2-9b-it",
        messages=conversation_history,
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )

    response_content = ""

    for chunk in completion:
        response_content += chunk.choices[0].delta.content or ""

    conversation_history.append({"role": "assistant", "content": response_content})

    # return message pairs as tuples
    return [
        (message["content"], conversation_history[i + 1]["content"])
        for i, message in enumerate(conversation_history[:-1])
        if message["role"] == "user"
    ]


# Function to generate a roadmap
def generate_roadmap(scenario):
    if not scenario.strip():
        return "Please provide your specific goal to generate the roadmap."

    messages = [
        {
            "role": "system",
            "content": """You are an AI mentor. Generate a roadmap in a structured format by splitting
            stages in different sections, each with a title + emoji, steps, and results to measure.
        """,
        },
        {"role": "user", "content": f"Generate a roadmap for: {scenario}"},
    ]

    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
    )
    return completion.choices[0].message.content


custom_theme = gr.themes.Soft(
    font=[gr.themes.GoogleFont("Lato"), "ui-sans-serif", "system-ui", "sans-serif"]
)


TITLE = """
<div style="display: flex; flex-wrap: nowrap; width: 100%; gap: 10px;">
    <img src="https://huggingface.co/spaces/its-deego/Purl/resolve/main/purl-logo.png" alt="Purl logo" width="60px" height="60px">
    <h1 src="font-size: 64px; margin-bottom: 10px;">Purl</h1>
</div>
"""

with gr.Blocks(theme=custom_theme) as demo:
    with gr.Row():
        gr.HTML(TITLE)
    with gr.Sidebar():
        gr.Markdown("## Settings", elem_classes="Title")
        language_input = gr.Dropdown(
            ["English", "French", "Spanish", "Mandarin", "Hindi", "German"],
            label="Language",
            interactive=True,
        )
        response_style_input = gr.Dropdown(
            ["Motivational", "Friendly", "Informative"],
            label="Response Style",
            interactive=True,
        )
    with gr.Tabs():
        with gr.TabItem("💬 Chat"):
            chatbot = gr.Chatbot(
                label="Chat History",
                show_copy_button=True,
                avatar_images=[None, "purl-logo.png"],
            )
            with gr.Row():
                user_input = gr.Textbox(
                    show_label="false",
                    placeholder="Enter your message...",
                    lines=2,
                )
            with gr.Row():
                send_button = gr.Button("Send Message 💬", variant="primary")

                def send_message(message, language, response_style):
                    updated_chat_history = chat_with_bot_stream(
                        message, language, response_style
                    )
                    return updated_chat_history, ""

                # Handle message response
                send_button.click(
                    send_message,
                    inputs=[user_input, language_input, response_style_input],
                    outputs=[chatbot, user_input],
                )

        with gr.TabItem("🎯 Generate Roadmap"):
            gr.Markdown("## Generate a Roadmap")
            scenario_input = gr.Textbox(label="Enter your goal and timeframe")
            generate_btn = gr.Button("Generate Roadmap")
            roadmap_output = gr.Textbox(label="Generated Roadmap", interactive=False)
            generate_btn.click(
                generate_roadmap, inputs=scenario_input, outputs=roadmap_output
            )

demo.launch()
