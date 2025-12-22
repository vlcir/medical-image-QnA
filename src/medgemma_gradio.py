import gradio as gr
import torch
from PIL import Image

import torch
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModelForImageTextToText
from datasets import load_dataset
import random

model_id = "unsloth/medgemma-4b-it"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model on {device}...")
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
).to(device)


def generate_answer(image_pil, question_text):
    """
    Gradio-compatible function: принимает изображение и вопрос, возвращает ответ.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"answer the given question as short as possible, do not give any explanation just straight answer in no more than 5 words, <question>: {question_text}"}
            ]
        }
    ]

    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    inputs = processor(
        text=text_prompt,
        images=image_pil.convert("RGB"),
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            num_beams=1,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    generated_text = processor.decode(generated_ids, skip_special_tokens=True)

    return generated_text.strip()


demo = gr.Interface(
    fn=generate_answer,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Textbox(label="Question", placeholder="Enter your question here...")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="🖼️ Visual Question Answering with MedGemma/PaliGemma",
    description="Upload an image and ask a question. The model will give a short answer (≤5 words).",
    examples=[
        ["example_image.jpg", "What is shown in the image?"],
        ["another_image.png", "Is there a tumor present?"]
    ],
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch(share=True)
