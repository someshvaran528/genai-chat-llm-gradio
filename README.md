## Development and Deployment of a 'Chat with LLM' Application Using the Gradio Blocks Framework

### AIM:
To design and deploy a "Chat with LLM" application by leveraging the Gradio Blocks UI framework to create an interactive interface for seamless user interaction with a large language model.

### PROBLEM STATEMENT:
Interacting directly with a Large Language Model through programming code can be difficult for users who are not familiar with programming. Therefore, a user-friendly chatbot interface is required to allow users to enter questions, view generated responses, maintain conversation history, and control the behaviour of the LLM. This project develops such an interactive application using Gradio Blocks.

### DESIGN STEPS:

#### STEP 1:
Load the Hugging Face API and create a Client to connect to the FalconLM text-generation model.

#### STEP 2:
Create the generate() function to take the user’s prompt and maximum token value and return the generated text.


#### STEP 3:
Build the Gradio interface with a prompt box, token slider, and completion output, then launch the app.

### PROGRAM:
```python
import os
import io
import IPython.display
from PIL import Image
import base64
import requests

requests.adapters.DEFAULT_TIMEOUT = 60

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

hf_api_key = os.environ["HF_API_KEY"]

import requests
import json
from text_generation import Client

client = Client(
    os.environ["HF_API_FALCOM_BASE"],
    headers={"Authorization": f"Basic {hf_api_key}"},
    timeout=120
)

prompt = "Has math been invented or discovered?"

client.generate(
    prompt,
    max_new_tokens=256
).generated_text

import gradio as gr

def generate(input, slider):
    output = client.generate(
        input,
        max_new_tokens=slider
    ).generated_text

    return output

demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Slider(
            label="Max new tokens",
            value=20,
            maximum=1024,
            minimum=1
        )
    ],
    outputs=[
        gr.Textbox(label="Completion")
    ]
)

gr.close_all()

demo.launch(
    share=True,
    server_port=int(os.environ["PORT1"])
)

import random

def respond(message, chat_history):
    bot_message = random.choice([
        "Tell me more about it",
        "Cool, but I'm not interested",
        "Hmmmm, ok then"
    ])

    chat_history.append(
        (message, bot_message)
    )

    return "", chat_history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height=240)

    msg = gr.Textbox(label="Prompt")

    btn = gr.Button("Submit")

    clear = gr.ClearButton(
        components=[msg, chatbot],
        value="Clear console"
    )

    btn.click(
        respond,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot]
    )

    msg.submit(
        respond,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot]
    )

gr.close_all()

demo.launch(
    share=True,
    server_port=int(os.environ["PORT2"])
)

def format_chat_prompt(message, chat_history):
    prompt = ""

    for turn in chat_history:
        user_message, bot_message = turn

        prompt = (
            f"{prompt}\n"
            f"User: {user_message}\n"
            f"Assistant: {bot_message}"
        )

    prompt = (
        f"{prompt}\n"
        f"User: {message}\n"
        f"Assistant:"
    )

    return prompt

def respond(message, chat_history):
    formatted_prompt = format_chat_prompt(
        message,
        chat_history
    )

    bot_message = client.generate(
        formatted_prompt,
        max_new_tokens=1024,
        stop_sequences=[
            "\nUser:",
            "<|endoftext|>"
        ]
    ).generated_text

    chat_history.append(
        (message, bot_message)
    )

    return "", chat_history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height=240)

    msg = gr.Textbox(label="Prompt")

    btn = gr.Button("Submit")

    clear = gr.ClearButton(
        components=[msg, chatbot],
        value="Clear console"
    )

    btn.click(
        respond,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot]
    )

    msg.submit(
        respond,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot]
    )

gr.close_all()

demo.launch(
    share=True,
    server_port=int(os.environ["PORT3"])
)

def format_chat_prompt(message, chat_history, instruction):
    prompt = f"System:{instruction}"

    for turn in chat_history:
        user_message, bot_message = turn

        prompt = (
            f"{prompt}\n"
            f"User: {user_message}\n"
            f"Assistant: {bot_message}"
        )

    prompt = (
        f"{prompt}\n"
        f"User: {message}\n"
        f"Assistant:"
    )

    return prompt

def respond(
    message,
    chat_history,
    instruction,
    temperature=0.7
):
    prompt = format_chat_prompt(
        message,
        chat_history,
        instruction
    )

    chat_history = chat_history + [
        [message, ""]
    ]

    stream = client.generate_stream(
        prompt,
        max_new_tokens=1024,
        stop_sequences=[
            "\nUser:",
            "<|endoftext|>"
        ],
        temperature=temperature
    )

    acc_text = ""

    for idx, response in enumerate(stream):
        text_token = response.token.text

        if response.details:
            return

        if idx == 0 and text_token.startswith(" "):
            text_token = text_token[1:]

        acc_text += text_token

        last_turn = list(
            chat_history.pop(-1)
        )

        last_turn[-1] += acc_text

        chat_history = chat_history + [
            last_turn
        ]

        yield "", chat_history

        acc_text = ""

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height=240)

    msg = gr.Textbox(label="Prompt")

    with gr.Accordion(
        label="Advanced options",
        open=False
    ):
        system = gr.Textbox(
            label="System message",
            lines=2,
            value=(
                "A conversation between a user and an "
                "LLM-based AI assistant. The assistant "
                "gives helpful and honest answers."
            )
        )

        temperature = gr.Slider(
            label="temperature",
            minimum=0.1,
            maximum=1,
            value=0.7,
            step=0.1
        )

    btn = gr.Button("Submit")

    clear = gr.ClearButton(
        components=[msg, chatbot],
        value="Clear console"
    )

    btn.click(
        respond,
        inputs=[
            msg,
            chatbot,
            system
        ],
        outputs=[
            msg,
            chatbot
        ]
    )

    msg.submit(
        respond,
        inputs=[
            msg,
            chatbot,
            system
        ],
        outputs=[
            msg,
            chatbot
        ]
    )

gr.close_all()

demo.queue().launch(
    share=True,
    server_port=int(os.environ["PORT4"])
)

gr.close_all()
```
### OUTPUT:
<img width="1236" height="226" alt="image" src="https://github.com/user-attachments/assets/49b03a8f-2e02-4628-bc01-38092f1bec23" />

<img width="1172" height="622" alt="image" src="https://github.com/user-attachments/assets/f878ebfc-7018-475b-bf5c-d28c9b77d509" />

<img width="1147" height="611" alt="image" src="https://github.com/user-attachments/assets/07472946-281b-44af-bc3f-6811cc6ab771" />

<img width="1137" height="615" alt="image" src="https://github.com/user-attachments/assets/b2c64c78-e9b1-490e-83de-e96bfd181587" />

<img width="1112" height="496" alt="image" src="https://github.com/user-attachments/assets/d9379899-fc18-44d6-af46-eb15d0e17d14" />
### RESULT:


Therefore,the program for Development and Deployment of a 'Chat with LLM' Application Using the Gradio Blocks Framework is executed successfully and the output is verified.
