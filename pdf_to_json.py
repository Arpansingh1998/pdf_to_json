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

            # Prepare the image and a text prompt for the model
            pixel_values = processor(image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device)

            # Create the prompt and its attention mask separately
            prompt_text = "<s_custom>"
            decoder_input_ids = processor.tokenizer(
                prompt_text, add_special_tokens=False, return_tensors="pt"
            ).input_ids

            # Generate the output from the model
            outputs = model.generate(
                pixel_values.to(device),
                decoder_input_ids=decoder_input_ids.to(device),
                max_length=model.decoder.config.max_position_embeddings,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
                # These parameters prevent repetitive outputs
                no_repeat_ngram_size=2,
                early_stopping=True,
            )

            # Decode the output to a string
            pred_string = processor.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

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