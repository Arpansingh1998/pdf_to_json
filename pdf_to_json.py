# infer_donut_fixed.py
import fitz
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
import re
import xml.etree.ElementTree as ET
import json

processor = DonutProcessor.from_pretrained("./donut-finetuned")
model = VisionEncoderDecoderModel.from_pretrained("./donut-finetuned")

processor = DonutProcessor.from_pretrained("./donut-finetuned")
model = VisionEncoderDecoderModel.from_pretrained("./donut-finetuned")



device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ensure config tokens are consistent
model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
model.config.eos_token_id = processor.tokenizer.eos_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id


def clean_output(raw: str) -> str:
    """
    Extract <s_custom>...</s_custom> block and ensure valid XML.
    """
    match = re.search(r"<s_custom>.*</s_custom>", raw, re.DOTALL)
    if not match:
        return ""

    block = match.group(0)

    # Try parsing with XML parser
    try:
        ET.fromstring(block)  # validates XML
        return block
    except ET.ParseError:
        # Simple fixes for common issues
        block = block.replace("&", "&amp;")  # escape bad ampersands
        try:
            ET.fromstring(block)
            return block
        except ET.ParseError:
            return ""


def process_pdf_with_donut(pdf_path):
    all_page_data = []
    pdf_document = fitz.open(pdf_path)

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

        outputs = model.generate(
            pixel_values,
            max_length=1024,  # more space for tags
            decoder_start_token_id=processor.tokenizer.bos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )

        decoded = processor.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Extract structured block
        cleaned = clean_output(decoded)

        page_result = {
            "page": page_num + 1,
            "data": {
                "raw_output": decoded,
                "ground_truth_like": cleaned if cleaned else None
            }
        }
        all_page_data.append(page_result)

    pdf_document.close()
    return {"document_data": all_page_data}


if __name__ == "__main__":
    out = process_pdf_with_donut("pdf11.pdf")
    print("Q" * 100)
    print(json.dumps(out, ensure_ascii=False, indent=2))
