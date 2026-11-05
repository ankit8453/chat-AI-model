"""Chat screen for the demo (Gradio): a public link straight from Colab."""
from __future__ import annotations

from typing import Callable

from demo.chat import answer

HANDED_OVER = "\n\n*Handed to human*"


def build_app(memory: dict, ask: Callable[[list[dict]], str]):
    import gradio as gr

    def respond(message: str, history: list[dict]) -> str:
        earlier = [{"role": m["role"], "content": m["content"].replace(HANDED_OVER, "")} for m in history]
        result = answer(memory, message, earlier, ask)
        print(f"[{result['reason']}] {message!r}")
        return result["text"] + (HANDED_OVER if result["needs_human"] else "")

    return gr.ChatInterface(respond, type="messages", title=memory["business_name"],
                            description="MBG AI demo: replies only from this business's memory.",
                            chatbot=gr.Chatbot(type="messages", height=520),
                            textbox=gr.Textbox(placeholder="Type a message...", submit_btn=True))
