from peft import PeftModel, PeftConfig
def get_clf_model():
    base_model = AutoModelForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=2,
        id2label=id2label,
        label2id={"Normal": 0, "Medical": 1},
        ignore_mismatched_sizes=True
    )

    model_inference = PeftModel.from_pretrained(base_model, "med_clf")
    model_inference.eval()

    return model_inference


def check_image(image, model):
    model.eval()
    device = next(model.parameters()).device
    image = image.convert("L").convert("RGB") # to Black-White

    inputs = image_processor(image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        probs = F.softmax(logits, dim=-1)
        predicted_class_id = probs.argmax().item()
        confidence = probs[0][predicted_class_id].item()

    label_map = {0: "Normal", 1: "Medical"}

    pred_label_name = label_map[predicted_class_id]
    
    return pred_label_name