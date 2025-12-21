import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart
from aiogram.types import FSInputFile
import nest_asyncio


def load_model():
    gc.collect()
    torch.cuda.empty_cache()
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
    model.qwen = apply_lora_to_qwen(model.qwen)
    
    checkpoint_path = "checkpoints/lora_epoch_1.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cuda:0")
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Please train the model first.")
    
    model.to("cuda:0").eval()
    return model, image_processor

# Telegram Bot Code with aiogram
TOKEN = ''  

model, image_processor = load_model()

bot = Bot(token=TOKEN)
dp = Dispatcher()

@dp.message(CommandStart())
async def start(message: types.Message):
    await message.reply('Send me an image with a caption as the question, and I\'ll answer using the VQA model!')

@dp.message()
async def handle_photo(message: types.Message):
    print("Received message from user:", message.from_user.id)
    print("Message has photo:", bool(message.photo))
    print("Message has caption:", bool(message.caption))
    if message.photo and message.caption:
        try:
            print("Starting photo processing...")
            # Download the photo
            photo = message.photo[-1]  # Get the highest resolution
            print("Selected photo size:", photo.width, "x", photo.height)
            file = await bot.get_file(photo.file_id)
            print("Got file info:", file.file_id, file.file_path)
            photo_path = 'temp_image.jpg'
            await bot.download_file(file.file_path, photo_path)
            print("Downloaded photo to:", photo_path)
            
            # Process the image
            print("Opening image...")
            image = Image.open(photo_path).convert("RGB")
            print("Resizing image...")
            image = image.resize((224, 224), Image.LANCZOS)
            print("Processing image with image_processor...")
            processed_image = image_processor(image, return_tensors="pt").pixel_values.to("cuda:0")
            print("Processed image shape:", processed_image.shape)
            
            # Get the question (caption)
            question = message.caption
            print("Question:", question)
            
            # Generate answer
            print("Generating answer...")
            prediction = generate_answer(model, processed_image, [question])[0]
            print("Generated prediction:", prediction)
            
            # Reply
            print("Sending reply...")
            await message.reply(f"Question: {question}\nAnswer: {prediction}")
            print("Reply sent.")
            
            # Clean up
            print("Cleaning up...")
            os.remove(photo_path)
            del processed_image
            torch.cuda.empty_cache()
            print("Cleanup complete.")
        except Exception as e:
            print("Error during processing:", str(e))
            await message.reply(f"Error processing photo: {str(e)}")
    else:
        print("Message does not have photo or caption. Sending reminder.")
        await message.reply('Please send an image with a caption as the question.')

async def main():
    await dp.start_polling(bot)

if __name__ == '__main__':
    nest_asyncio.apply()
    asyncio.run(main())
