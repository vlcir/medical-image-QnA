import os
import gc
import torch
import torch.nn as nn
import warnings
import logging

from PIL import Image
from torch.utils.data import DataLoader

from transformers import CLIPVisionModel, AutoImageProcessor
from transformers import AutoTokenizer, AutoModelForCausalLM



from peft import LoraConfig, get_peft_model, TaskType

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback



from utils import (
    apply_lora_to_qwen,
    load_qwen_frozen,
    save_checkpoint,
    generate_answer,
    evaluate_bert_f1_and_examples,
    BertF1Callback,
    CheckpointEveryEpoch
)

from visual_model import VisionMLP
from MultiModalVQA import MultiModalVQA, LitMultiModalVQA


warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
logging.getLogger('transformers').setLevel(logging.ERROR)


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    print("Loading models...")
    tokenizer, qwen = load_qwen_frozen()
    vision_model, image_processor = get_model(
        encoder_choice="pubmedclip",
        mlp_output_dim=qwen.config.hidden_size,
        hidden_dim=2048
    )
    model = MultiModalVQA(
        vision_mlp_model=vision_model,
        qwen=qwen,
        tokenizer=tokenizer
    )

    train_loader = get_dataloader("vqa_rad", image_processor, batch_size=16)
    test_loader = get_dataloader("vqa_rad", image_processor, batch_size=1, max_samples=20)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-{epoch}-{val_loss:.4f}",
        save_weights_only=False
    )

    # ----------------- MLP Stage -----------------
    print("\nStarting MLP training...")
    lit_model = LitMultiModalVQA(
        model=model,
        lr=3e-5,
        stage="mlp"
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        max_epochs=3,
        callbacks=[checkpoint_callback, BertF1Callback(test_loader), CheckpointEveryEpoch("mlp")],
        log_every_n_steps=10,
    )

    trainer.fit(
        lit_model,
        train_dataloaders=train_loader,
        val_dataloaders=test_loader,
    )
    model_mlp = lit_model.model
    model_mlp.to("cuda:0")
    evaluate_bert_f1_and_examples(model_mlp, test_loader, device="cuda:0", num_examples=3)

    # ----------------- LoRA Stage -----------------
    print("\nApplying LoRA to Qwen...")
    model.qwen = apply_lora_to_qwen(model.qwen)

    lit_model_lora = LitMultiModalVQA(
        model=model,
        lr=1e-4,
        stage="lora"
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        max_epochs=2,
        callbacks=[checkpoint_callback, BertF1Callback(test_loader), CheckpointEveryEpoch("lora")],
        log_every_n_steps=10,
    )

    trainer.fit(
        lit_model_lora,
        train_dataloaders=train_loader,
        val_dataloaders=test_loader,
    )

    model_lora = lit_model_lora.model
    model_lora.to("cuda:0")
    evaluate_bert_f1_and_examples(model_lora, test_loader, device="cuda:0", num_examples=3)
