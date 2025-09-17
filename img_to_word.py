from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

processor = DonutProcessor.from_pretrained("./donut-finetuned")
model = VisionEncoderDecoderModel.from_pretrained("./donut-finetuned")

image = Image.open("pdf_images/page_1.png").convert("RGB")
task_prompt = "<s_custom>"

inputs = processor(image, task_prompt, return_tensors="pt")
model.to("cuda" if torch.cuda.is_available() else "cpu")

outputs = model.generate(**inputs, max_length=1024)
result = processor.batch_decode(outputs, skip_special_tokens=True)[0]

print("Extracted JSON:", result)