# infer_donut_fixed.py
import fitz
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
import json

processor = DonutProcessor.from_pretrained("./donut-finetuned")
model = VisionEncoderDecoderModel.from_pretrained("./donut-finetuned")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ensure config tokens are consistent (in case)
model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
model.config.eos_token_id = processor.tokenizer.eos_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id

def process_pdf_with_donut(pdf_path):
    all_page_data = []
    pdf_document = fitz.open(pdf_path)

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

        # Use generate() with decoder_start_token_id and eos_token_id
        outputs = model.generate(
            pixel_values,
            max_length=512,
            decoder_start_token_id=processor.tokenizer.bos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )

        decoded = processor.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Try to load JSON; if fails, print diagnostics
        try:
            page_json = json.loads(decoded)
        except json.JSONDecodeError as e:
            print("❗ JSON decode failed for page", page_num+1)
            print("Decoded string (raw):\n", decoded[:1000])
            # print token-level debug info
            token_ids = outputs[0].tolist()
            print("Token ids:", token_ids[:40], "...")
            tokens = processor.tokenizer.convert_ids_to_tokens(token_ids)[:60]
            print("Tokens:", tokens, "...")
            page_json = {"raw_output": decoded}
        all_page_data.append({"page": page_num + 1, "data": page_json})

    pdf_document.close()
    return {"document_data": all_page_data}

if __name__ == "__main__":
    out = process_pdf_with_donut("pdf11.pdf")
    print(json.dumps(out, ensure_ascii=False, indent=2))
