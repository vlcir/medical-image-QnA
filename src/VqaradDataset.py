class VqaradDataset(Dataset):
    def __init__(self, image_processor, split='train', hf_repo_id='flaviagiammarino/vqa-rad'):
        self.image_processor = image_processor
        self.dataset = load_dataset(hf_repo_id, split=split, streaming=False)
        self.dataset = list(self.dataset)  # для небольшого датасета допустимо

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image'].convert("RGB")
        processed_output = self.image_processor(image, return_tensors="pt")

        if hasattr(processed_output, 'pixel_values'):
            processed_image = processed_output.pixel_values
        else:
            processed_image = processed_output

        if processed_image.dim() == 4:
            processed_image = processed_image.squeeze(0)

        return {
            "image": processed_image,
            "question": item['question'],
            "answer": item['answer']
        }
