import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from pytorch_lightning.callbacks import Callback
import gc


def apply_lora_to_qwen(
    qwen,
    r=8,
    alpha=16,
    dropout=0.05
):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ]
    )

    qwen = get_peft_model(qwen, lora_config)
    qwen.print_trainable_parameters()
    return qwen

def load_qwen_frozen():
    model_name = "Qwen/Qwen1.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    qwen = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cuda:0",
        load_in_8bit=False,
        low_cpu_mem_usage=True
    )
    qwen.config.pad_token_id = tokenizer.pad_token_id
    for p in qwen.parameters():
        p.requires_grad = False
    qwen.gradient_checkpointing_enable()
    qwen.enable_input_require_grads()
    return tokenizer, qwen

def save_checkpoint(model, epoch, checkpoint_dir="checkpoints", stage="mlp"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{stage}_epoch_{epoch}.pt")
    
    if stage == "mlp":
        torch.save({
            'epoch': epoch,
            'vision_mlp_state_dict': model.vision.state_dict(),
            'model_state_dict': model.state_dict(),
        }, checkpoint_path)
    else:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
        }, checkpoint_path)
    
    print(f"Checkpoint saved: {checkpoint_path}")

class BertF1Callback(Callback):
    def __init__(self, test_dataloader, device="cuda:0", num_examples=3):
        super().__init__()
        self.test_dataloader = test_dataloader
        self.device = device
        self.num_examples = num_examples

    @torch.no_grad()
    def on_epoch_end(self, trainer, pl_module):
        pl_module.eval()
        model = pl_module.model
        preds, refs, questions = [], [], []

        for batch_idx, batch in enumerate(self.test_dataloader):
            images = batch["image"].to(self.device)
            batch_questions = batch["question"]
            batch_answers = batch["answer"]
            batch_preds = generate_answer(model, images, batch_questions, device=self.device)
            preds.extend(batch_preds)
            refs.extend(batch_answers)
            questions.extend(batch_questions)
            if len(preds) >= self.num_examples:
                break

        print("\nSample predictions (Epoch {}):".format(trainer.current_epoch + 1))
        for i in range(min(self.num_examples, len(preds))):
            print(f"\n  Example {i + 1}:")
            print(f"    Question:   {questions[i]}")
            print(f"    Predicted:  {preds[i]}")
            print(f"    Reference:  {refs[i]}")

        pl_module.train()

class CheckpointEveryEpoch(Callback):
    def __init__(self, stage):
        super().__init__()
        self.stage = stage

    def on_train_epoch_end(self, trainer, pl_module):
        save_checkpoint(pl_module.model, trainer.current_epoch + 1, stage=self.stage)



@torch.no_grad()
def generate_answer(model, images, questions, max_length=32, device="cuda:0"):
    model.eval()
    batch_size = images.size(0)
    results = []

    base_model = model.get_base_model()

    for i in range(batch_size):
        single_image = images[i:i+1]

        single_image = single_image.to(device=device, dtype=next(model.vision.parameters()).dtype)
        if torch.isnan(single_image).any():
            single_image = torch.nan_to_num(single_image, nan=0.0)

        patch_embeds = model.vision(single_image)
        patch_embeds = patch_embeds.to(model.qwen.dtype).to(device)
        if torch.isnan(patch_embeds).any():
            patch_embeds = torch.nan_to_num(patch_embeds, nan=0.0)

        q_tok = model.tokenizer(
            [questions[i]],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64
        ).to(device)

        text_embeds = base_model.get_input_embeddings()(q_tok.input_ids).to(model.qwen.dtype)
        inputs_embeds = torch.cat([patch_embeds, text_embeds], dim=1)
        patch_mask = torch.ones((1, patch_embeds.size(1)), device=device)
        attention_mask = torch.cat([patch_mask, q_tok.attention_mask], dim=1)

        out_ids = model.qwen.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=7,
            min_new_tokens=1,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.5,
            no_repeat_ngram_size=2,
            eos_token_id=model.tokenizer.eos_token_id,
            pad_token_id=model.tokenizer.pad_token_id
        )

        text = model.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        if text.startswith(questions[i]):
            text = text[len(questions[i]):].strip()
        text = text.split('.')[0].split('!')[0].split('?')[0].strip()
        if not text:
            text = "unknown"

        results.append(text)

        del patch_embeds, text_embeds, inputs_embeds, attention_mask, out_ids, q_tok
        torch.cuda.empty_cache()

    return results

@torch.no_grad()
def evaluate_bert_f1_and_examples(model, dataloader, device="cuda:0", num_examples=3):
    model = model.to(device)  # Ensure model is on the correct device
    model.eval()
    preds, refs, questions = [], [], []

    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        batch_questions = batch["question"]
        batch_answers = batch["answer"]

        batch_preds = generate_answer(model, images, batch_questions, device=device)
        preds.extend(batch_preds)
        refs.extend(batch_answers)
        questions.extend(batch_questions)

        if len(preds) >= num_examples:
            break

    print("\nSample predictions:")
    for i in range(min(num_examples, len(preds))):
        print(f"\n  Example {i + 1}:")
        print(f"    Question:   {questions[i]}")
        print(f"    Predicted:  {preds[i]}")
        print(f"    Reference:  {refs[i]}")

    return 0.0


