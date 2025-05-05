import os
import openai
import gradio as gr

# ── Config via env ────────────────────────────────────────────────────────────
openai.api_key = os.environ.get("OPENAI_API_KEY")       # required
MODEL           = os.getenv("OPENAI_MODEL", "gpt-4o")
SYSTEM_PROMPT   = os.getenv(                       # keeps the demo simple
    "SYSTEM_PROMPT",
    "You are a helpful assistant named PurpLLM."
)

# ── Chat handler ─────────────────────────────────────────────────────────────
def chat(user_input: str, chat_history: list[tuple[str, str]]):
    """Gradio expects (updated_history, cleared_input)"""
    # build messages array: system + full history + latest user message
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for human, bot in chat_history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": bot})
    messages.append({"role": "user", "content": user_input})

    # call OpenAI
    response = openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.7,
    )
    reply = response.choices[0].message.content
    chat_history.append((user_input, reply))
    return chat_history, ""        # clears input box

# ── Gradio UI skeleton ───────────────────────────────────────────────────────
with gr.Blocks(title="PurpLLM Chatbot") as ui:
    gr.Markdown("## 🟣 PurpLLM Chatbot &nbsp;—&nbsp; OpenAI demo")
    chatbot = gr.Chatbot()
    userbox = gr.Textbox(placeholder="Type a message …", show_label=False)
    userbox.submit(chat, inputs=[userbox, chatbot], outputs=[chatbot, userbox])
    gr.Button("Send").click(chat, [userbox, chatbot], [chatbot, userbox])

def main():
    ui.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()
