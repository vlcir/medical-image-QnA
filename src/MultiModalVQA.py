import torch
import torch.nn as nn
import pytorch_lightning as pl
from peft import LoraConfig, get_peft_model, TaskType
from transformers import CLIPVisionModel, AutoImageProcessor
from transformers import AutoTokenizer, AutoModelForCausalLM

class MultiModalVQA(nn.Module):
    def __init__(self, vision_mlp_model, qwen, tokenizer):
        super().__init__()
        self.vision = vision_mlp_model
        self.qwen = qwen
        self.tokenizer = tokenizer
        self.device = torch.device("cuda:0")

    def get_base_model(self):
        if hasattr(self.qwen, "get_input_embeddings"):
            return self.qwen
        if hasattr(self.qwen, "base_model"):
            return self.qwen.base_model
        raise RuntimeError("Cannot locate base language model")

    def forward(self, images, questions, answers=None):
        device = self.device
        dtype = self.qwen.dtype
        patch_embeds = self.vision(images)
        if torch.isnan(patch_embeds).any():
            patch_embeds = torch.nan_to_num(patch_embeds, nan=0.0)
        patch_embeds = patch_embeds.to(dtype).to(device)
        B, P, H = patch_embeds.shape
        q_tok = self.tokenizer(
            questions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64
        ).to(device)
        

        base_model = self.get_base_model()
        text_embeds = base_model.get_input_embeddings()(q_tok.input_ids)

        
        if torch.isnan(text_embeds).any():
            text_embeds = torch.nan_to_num(text_embeds, nan=0.0)
        text_mask = q_tok.attention_mask.to(device)
        inputs_embeds = torch.cat([patch_embeds, text_embeds], dim=1)
        patch_mask = torch.ones((B, P), device=device, dtype=text_mask.dtype)
        attention_mask = torch.cat([patch_mask, text_mask], dim=1)
        labels = None
        if answers is not None:
            a_tok = self.tokenizer(
                answers,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=32
            ).to(device)
            labels_list = []
            for i in range(B):
                vis_pad = torch.full((P,), -100, device=device, dtype=torch.long)
                q_len = text_mask[i].sum().item()
                q_pad = torch.full((q_len,), -100, device=device, dtype=torch.long)
                ans_len = (a_tok.input_ids[i] != self.tokenizer.pad_token_id).sum().item()
                ans_tokens = a_tok.input_ids[i, :ans_len]
                label_seq = torch.cat([vis_pad, q_pad, ans_tokens], dim=0)
                labels_list.append(label_seq)
            max_len = max(len(l) for l in labels_list)
            labels = torch.full((B, max_len), -100, device=device, dtype=torch.long)
            for i, label_seq in enumerate(labels_list):
                labels[i, :len(label_seq)] = label_seq
            a_embeds = base_model.get_input_embeddings()(a_tok.input_ids).to(dtype).to(device)
            if torch.isnan(a_embeds).any():
                a_embeds = torch.nan_to_num(a_embeds, nan=0.0)
            a_mask = (a_tok.input_ids != self.tokenizer.pad_token_id).long().to(device)
            inputs_embeds = torch.cat([inputs_embeds, a_embeds], dim=1)
            attention_mask = torch.cat([attention_mask, a_mask], dim=1)
            if labels.size(1) != inputs_embeds.size(1):
                if labels.size(1) > inputs_embeds.size(1):
                    pad_len = labels.size(1) - inputs_embeds.size(1)
                    pad_emb = torch.zeros(B, pad_len, H, device=device, dtype=dtype)
                    inputs_embeds = torch.cat([inputs_embeds, pad_emb], dim=1)
                    pad_mask = torch.zeros(B, pad_len, device=device, dtype=attention_mask.dtype)
                    attention_mask = torch.cat([attention_mask, pad_mask], dim=1)
                elif inputs_embeds.size(1) > labels.size(1):
                    pad_len = inputs_embeds.size(1) - labels.size(1)
                    pad_labels = torch.full((B, pad_len), -100, device=device, dtype=torch.long)
                    labels = torch.cat([labels, pad_labels], dim=1)
        if torch.isnan(inputs_embeds).any():
            inputs_embeds = torch.nan_to_num(inputs_embeds, nan=0.0)
        out = self.qwen(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        return out

class LitMultiModalVQA(pl.LightningModule):
    def __init__(
        self,
        model: MultiModalVQA,
        lr: float,
        stage: str = "mlp"
    ):
        super().__init__()
        self.add_module("model", model)  # Register as submodule
        self.lr = lr
        self.stage = stage
        self.save_hyperparameters(ignore=["model"])

    def forward(self, images, questions, answers=None):
        return self.model(images, questions, answers)

    def training_step(self, batch, batch_idx):
        images = batch["image"]
        questions = batch["question"]
        answers = batch["answer"]

        out = self(images, questions, answers)
        loss = out.loss

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=images.size(0)
        )
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["image"]
        questions = batch["question"]
        answers = batch["answer"]

        out = self(images, questions, answers)
        loss = out.loss

        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            batch_size=images.size(0)
        )
        return loss

    def configure_optimizers(self):
        if self.stage == "mlp":
            params = self.model.vision.mlp.parameters()
        else:
            params = self.model.qwen.parameters()

        optimizer = torch.optim.AdamW(
            params,
            lr=self.lr,
            weight_decay=0.01
        )
        return optimizer

  
