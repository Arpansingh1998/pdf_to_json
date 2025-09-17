import fitz
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
import json

# -----------------------------
# Step 1: Load fine-tuned model and processor
# -----------------------------
try:
    processor = DonutProcessor.from_pretrained("./donut-finetuned")
    model = VisionEncoderDecoderModel.from_pretrained("./donut-finetuned")
except OSError:
    print("Error: The trained model and processor were not found.")
    print("Please ensure you have run the training script and the files are in './donut-finetuned'.")
    exit()

# Move the model to a GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# -----------------------------
# Step 2: Define PDF processing function
# -----------------------------
def process_pdf_with_donut(pdf_path):
    """
    Converts a PDF to images, processes each image with the Donut model,
    and returns a combined JSON output.
    """
    all_page_data = []

    try:
        pdf_document = fitz.open(pdf_path)

        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            print(f"Processing page {page_num + 1}...")

            # Prepare the image and get the attention mask
            # Pass a prompt to guide the model's generation (recommended for fine-tuned models)
            inputs = processor(
                image, 
                text="<s_data_extraction>", # You may need to change this prompt to match your training
                return_tensors="pt"
            )

            # Explicitly extract both tensors and move them to the device
            pixel_values = inputs["pixel_values"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            # Generate the output, making sure to pass the attention_mask
            outputs = model.generate(
                pixel_values,
                attention_mask=attention_mask,
                max_length=256,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id
            )

            pred_string = processor.batch_decode(outputs, skip_special_tokens=True)[0]

            try:
                page_json = json.loads(pred_string)
                print("✅ Extracted JSON successfully.")
            except json.JSONDecodeError:
                print("⚠️ Could not decode output as JSON. Storing as a string.")
                page_json = {"raw_output": pred_string}
            
            all_page_data.append({"page": page_num + 1, "data": page_json})

        pdf_document.close()

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    return {"document_data": all_page_data}


# -----------------------------
# Step 3 & 4: Run the process and print output
# -----------------------------
pdf_file_path = "ram.pdf"
final_json_output = process_pdf_with_donut(pdf_file_path)

if final_json_output:
    print(json.dumps(final_json_output, ensure_ascii=False, indent=4))
    print("\nJSON data printed to the console.")
else:
    print("No data was extracted from the PDF.")