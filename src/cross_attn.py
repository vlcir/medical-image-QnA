class SwiGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_, gate = x.chunk(2, dim=-1)
        return f.silu(gate) * x_


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, scale_base: int = 512, use_xpos: bool = True) -> None:
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.use_xpos = use_xpos
        self.scale_base = scale_base
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.register_buffer('scale', scale)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        if not self.use_xpos:
            return freqs, torch.ones(1, device=device)
        power = (t - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim=-1)
        return freqs, scale


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos: torch.Tensor, t: torch.Tensor, scale: float = 1.) -> torch.Tensor:
    return (t * pos.cos() * scale) + (rotate_half(t) * pos.sin() * scale)


def l2norm(t: torch.Tensor) -> torch.Tensor:
    return f.normalize(t, dim=-1)

class CrossAttentionBlock(nn.Module):
    """
    Cross-Attention: Query от изображения, Key/Value от текста
    """
    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8, dropout: float = 0.2, forward_expansion: int = 2) -> None:
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.embed_dim = heads * dim_head
        
        # Проекции для Q, K, V
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        
        # Масштабирующие параметры для QK-normalization
        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))
        
        # Rotary embeddings для Query (image patches)
        self.rotary_emb = RotaryEmbedding(dim_head)
        
        # Output projection
        self.to_out = nn.Linear(dim_head * heads, dim)
        
        # Normalization
        self.norm = nn.LayerNorm(dim)
        
        # Feed-forward с SwiGLU
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, forward_expansion * dim * 2),  # *2 для SwiGLU
            SwiGLU(),
            nn.Dropout(dropout),
            nn.Linear(forward_expansion * dim, dim),
        )
        self.ff_norm = nn.LayerNorm(dim)

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_features: (B, num_patches, dim) - Query
            text_features: (B, seq_len_text, dim) - Key, Value
        Returns:
            output: (B, num_patches, dim)
        """
        B, num_patches, dim = image_features.shape
        _, seq_len_text, _ = text_features.shape
        device = image_features.device
        
        # Проекции
        q = self.to_q(image_features)  # (B, num_patches, heads * dim_head)
        k = self.to_k(text_features)    # (B, seq_len_text, heads * dim_head)
        v = self.to_v(text_features)    # (B, seq_len_text, heads * dim_head)
        
        # Reshape для multi-head attention
        q = q.reshape(B, num_patches, self.heads, self.dim_head).permute(0, 2, 1, 3)  # (B, heads, num_patches, dim_head)
        k = k.reshape(B, seq_len_text, self.heads, self.dim_head).permute(0, 2, 1, 3)  # (B, heads, seq_len_text, dim_head)
        v = v.reshape(B, seq_len_text, self.heads, self.dim_head).permute(0, 2, 1, 3)  # (B, heads, seq_len_text, dim_head)
        
        # QK Normalization
        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale
        
        # Rotary embeddings только для Query (image patches)
        positions_q, scale_q = self.rotary_emb(num_patches, device)
        q = apply_rotary_pos_emb(positions_q, q, scale_q)
        
        # Scaled dot-product attention
        attn_output = f.scaled_dot_product_attention(q, k, v, dropout_p=0.0 if not self.training else 0.1)
        
        # Reshape обратно
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, num_patches, self.embed_dim)
        
        # Output projection
        attn_output = self.to_out(attn_output)
        
        # Residual + Norm
        attn_output = self.norm(attn_output + image_features)
        
        # Feed-forward
        ff_output = self.feed_forward(attn_output)
        output = self.ff_norm(ff_output + attn_output)
        
        return output

class VisionCrossAttention(nn.Module):
    """
    Модель с Cross-Attention между image patches и text embeddings
    Выход: (B, num_patches, mlp_output_dim) - как у VisionMLP
    """
    def __init__(
        self,
        vision_encoder,
        encoder_output_dim: int,
        mlp_output_dim: int,
        text_embed_dim: int = 768,  # Размерность text embeddings (для Qwen токенайзера)
        hidden_dim: int = 2560,
        num_cross_attn_layers: int = 2,
        dim_head: int = 64,
        heads: int = 8,
        dropout: float = 0.2
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        
        # Проекция изображения к hidden_dim
        self.image_projection = nn.Linear(encoder_output_dim, hidden_dim)
        
        # Проекция текста к hidden_dim
        self.text_projection = nn.Linear(text_embed_dim, hidden_dim)
        
        # Несколько слоев Cross-Attention
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionBlock(
                dim=hidden_dim,
                dim_head=dim_head,
                heads=heads,
                dropout=dropout,
                forward_expansion=2
            ) for _ in range(num_cross_attn_layers)
        ])
        
        # Финальная проекция к mlp_output_dim
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, mlp_output_dim),
            nn.LayerNorm(mlp_output_dim)
        )

    def forward(self, pixel_values: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, C, H, W) - изображения
            text_embeddings: (B, seq_len_text, text_embed_dim) - эмбеддинги текста
        Returns:
            output: (B, num_patches, mlp_output_dim)
        """
        # Извлекаем патчи из изображения
        outputs = self.vision_encoder(pixel_values, output_hidden_states=False)
        patch_tokens = outputs.last_hidden_state[:, 1:, :]  # (B, num_patches, encoder_output_dim)
        
        # Проекции
        image_features = self.image_projection(patch_tokens)  # (B, num_patches, hidden_dim)
        text_features = self.text_projection(text_embeddings)  # (B, seq_len_text, hidden_dim)
        
        # Применяем Cross-Attention слои
        cross_features = image_features
        for layer in self.cross_attention_layers:
            cross_features = layer(cross_features, text_features)  # (B, num_patches, hidden_dim)
        
        # Финальная проекция
        output = self.output_projection(cross_features)  # (B, num_patches, mlp_output_dim)
        
        return output

def get_model_with_cross_attention(
    encoder_choice: str,
    mlp_output_dim: int,
    text_embed_dim: int = 768,
    hidden_dim: int = 2560,
    num_cross_attn_layers: int = 2
):
    """
    Создает модель с Cross-Attention
    
    Args:
        encoder_choice: 'standard_clip' или 'pubmedclip'
        mlp_output_dim: выходная размерность (например, 4096 для Qwen)
        text_embed_dim: размерность текстовых эмбеддингов
        hidden_dim: внутренняя размерность для Cross-Attention
        num_cross_attn_layers: количество слоев Cross-Attention
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_paths = {
        'standard_clip': 'openai/clip-vit-base-patch32',
        'pubmedclip': 'flaviagiammarino/pubmed-clip-vit-base-patch32',
    }
    
    from transformers import CLIPVisionModel, AutoImageProcessor
    
    vision_encoder = CLIPVisionModel.from_pretrained(model_paths[encoder_choice])
    image_processor = AutoImageProcessor.from_pretrained(model_paths[encoder_choice])
    
    encoder_output_dim = vision_encoder.config.hidden_size
    
    model = VisionCrossAttention(
        vision_encoder=vision_encoder,
        encoder_output_dim=encoder_output_dim,
        mlp_output_dim=mlp_output_dim,
        text_embed_dim=text_embed_dim,
        hidden_dim=hidden_dim,
        num_cross_attn_layers=num_cross_attn_layers
    ).to(device)
    
    # Замораживаем CLIP encoder
    for param in model.vision_encoder.parameters():
        param.requires_grad = False
    
    return model, image_processor


def test_cross_attention_model():
    """
    Тест модели с фейковыми текстовыми эмбеддингами
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("\n=== ТЕСТ CROSS-ATTENTION МОДЕЛИ ===\n")
    
    # Параметры
    encoder_choice = 'pubmedclip'
    mlp_output_dim = 4096
    text_embed_dim = 768
    batch_size = 4
    seq_len_text = 32  # длина текстовой последовательности
    
    # Создаем модель
    model, image_processor = get_model_with_cross_attention(
        encoder_choice=encoder_choice,
        mlp_output_dim=mlp_output_dim,
        text_embed_dim=text_embed_dim,
        hidden_dim=2560,
        num_cross_attn_layers=2
    )
    
    # Создаем фейковые данные
    # Изображения
    fake_images = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # Текстовые эмбеддинги (например, от Qwen токенайзера)
    fake_text_embeddings = torch.randn(batch_size, seq_len_text, text_embed_dim).to(device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(fake_images, fake_text_embeddings)
    
    print(f"✓ Форма входных изображений: {fake_images.shape}")
    print(f"✓ Форма текстовых эмбеддингов: {fake_text_embeddings.shape}")
    print(f"✓ Форма выходных патч-эмбеддингов: {output.shape}")
    print(f"✓ Ожидаемая форма: ({batch_size}, 49, {mlp_output_dim})")
    
    # Проверки
    assert output.shape == (batch_size, 49, mlp_output_dim), f"Неверная форма! Получено: {output.shape}"
    assert not torch.isnan(output).any(), "Обнаружены NaN в выходе!"
    assert not torch.isinf(output).any(), "Обнаружены Inf в выходе!"
    
    print(f"\n✓ Первый патч первого изображения (первые 10 значений):")
    print(output[0, 0, :10])
    
    print("\n=== ✓ ВСЕ ТЕСТЫ ПРОЙДЕНЫ ===\n")
    
    # Подсчет параметров
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Всего параметров: {total_params:,}")
    print(f"Обучаемых параметров: {trainable_params:,}")
    print(f"Замороженных параметров: {total_params - trainable_params:,}")


if __name__ == "__main__":
    test_cross_attention_model()
