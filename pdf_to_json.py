import fitz  # PyMuPDF
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
        # Open the PDF file
        pdf_document = fitz.open(pdf_path)

        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)

            # Convert page to a high-resolution PIL image
            pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72)) # Set DPI to 300 for higher quality
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            print(f"Processing page {page_num + 1}...")

            # Prepare the image for the model with a prompt to guide generation
            # This is crucial for telling the model what kind of output to produce.
            # Use the same prompt token you used during training.
            prompt = "<s_data_extraction>"
            
            inputs = processor(
                image, 
                text=prompt,
                return_tensors="pt"
            )

            # Explicitly pass attention_mask to avoid the warning and ensure proper generation
            pixel_values = inputs["pixel_values"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            # Generate the output from the model
            outputs = model.generate(
                pixel_values,
                attention_mask=attention_mask,
                max_length=256,
                # It's good practice to set these for reliable generation
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id
            )

            # Decode the output to a string
            pred_string = processor.batch_decode(outputs, skip_special_tokens=True)[0]

            # Try to parse the string as JSON
            try:
                page_json = json.loads(pred_string)
                print("✅ Extracted JSON successfully.")
            except json.JSONDecodeError:
                print("⚠️ Could not decode output as JSON. Storing as a string.")
                # This is a key line: if the model fails, we want to see the *text* it did generate
                page_json = {"raw_output": pred_string}
            
            # Append the result for this page
            all_page_data.append({"page": page_num + 1, "data": page_json})

        # Close the PDF document
        pdf_document.close()

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    # Return the combined JSON data
    return {"document_data": all_page_data}


# -----------------------------
# Step 3 & 4: Run the process and print output
# -----------------------------
pdf_file_path = "ram.pdf"  # 📌 Change this to your PDF file path
final_json_output = process_pdf_with_donut(pdf_file_path)

if final_json_output:
    print(json.dumps(final_json_output, ensure_ascii=False, indent=4))
    print("\nJSON data printed to the console.")
else:
    print("No data was extracted from the PDF.")