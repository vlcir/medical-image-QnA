class SlakeDataset(Dataset):
    def __init__(self, image_processor, split='train', hf_repo_id='BoKelvin/SLAKE'):
        self.image_processor = image_processor
        self.root_dir = snapshot_download(repo_id=hf_repo_id, repo_type='dataset')
        img_dir_path = os.path.join(self.root_dir, 'imgs')
        if not os.path.exists(img_dir_path):
            os.system(f"unzip -q -o {os.path.join(self.root_dir, 'imgs.zip')} -d {self.root_dir}")
        json_path = os.path.join(self.root_dir, f"{split}.json")
        with open(json_path, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
        self.dataset = [item for item in full_data if item['q_lang'] == 'en']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image_name = item['img_name']
        image_name = image_name.split('/')
        image_name = os.path.join(image_name[0], image_name[1])
        image_path = os.path.join(self.root_dir, 'imgs', image_name)
        image = Image.open(image_path).convert("RGB")
        
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
