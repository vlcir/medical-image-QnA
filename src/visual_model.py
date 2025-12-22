class VisionMLP(nn.Module):
    def __init__(self, vision_encoder, encoder_output_dim, mlp_output_dim, hidden_dim=2560):
        super().__init__()
        self.vision_encoder = vision_encoder
        
        self.mlp = nn.Sequential(
            nn.Linear(encoder_output_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, mlp_output_dim)
        )

        self.mlp.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            torch.nn.init.constant_(m.weight, 1.0)
            torch.nn.init.constant_(m.bias, 0.0)

    def forward(self, pixel_values):
        outputs = self.vision_encoder(pixel_values, output_hidden_states=False)
        patch_tokens = outputs.last_hidden_state[:, 1:, :]
        return self.mlp(patch_tokens)


def get_model(encoder_choice, mlp_output_dim, hidden_dim=2560):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_paths = {
        'standard_clip': 'openai/clip-vit-base-patch32',
        'pubmedclip': 'flaviagiammarino/pubmed-clip-vit-base-patch32',
        'large_clip': 'openai/clip-vit-large-patch14'
    }

    vision_encoder = CLIPVisionModel.from_pretrained(model_paths[encoder_choice])
    image_processor = AutoImageProcessor.from_pretrained(model_paths[encoder_choice])
    
    encoder_output_dim = vision_encoder.config.hidden_size

    model = VisionMLP(
        vision_encoder=vision_encoder,
        encoder_output_dim=encoder_output_dim,
        mlp_output_dim=mlp_output_dim
    ).to(device)

    for param in model.vision_encoder.parameters():
        param.requires_grad = False

    return model, image_processor


def get_dataloader(dataset_choice, image_processor, batch_size=4):
    if dataset_choice == 'slake':
        dataset = SlakeDataset(image_processor=image_processor, split='train')
    elif dataset_choice == 'vqa_rad':
        dataset = VqaradDataset(image_processor=image_processor, split='train')

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return dataloader


def eval(encoder_choice, dataset_choice, mlp_output_dim=4096):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n--- Обработка батча с ПАТЧАМИ ---")
    model, image_processor = get_model(encoder_choice=encoder_choice, mlp_output_dim=mlp_output_dim)
    dataloader = get_dataloader(dataset_choice=dataset_choice, image_processor=image_processor, batch_size=4)

    model.eval()
    with torch.no_grad():
        batch = next(iter(dataloader))
        images = batch['image'].to(device)
        image_embeddings = model(images)

        print(f"\n- Форма изображений: {images.shape}")
        print(f"- Форма патч-эмбеддингов: {image_embeddings.shape}")

        assert image_embeddings.dim() == 3, "Ожидалась 3D-форма для патчей!"
        assert image_embeddings.shape[-1] == mlp_output_dim, "Неверная размерность эмбеддинга!"
        

        print(f"- Патчей на изображение: {image_embeddings.shape[1]} (ожидалось ~49)")
        print(image_embeddings[0][0])
    
    print("\n*** УСПЕХ: Получены эмбеддинги ВСЕХ ПАТЧЕЙ ***")
