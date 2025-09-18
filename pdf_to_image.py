from pdf2image import convert_from_path
import os

def pdf_to_images(pdf_path, pdf_file, output_folder="val_images", dpi=300):
    os.makedirs(output_folder, exist_ok=True)
    images = convert_from_path(pdf_path, dpi=dpi)
    name_without_ext = os.path.splitext(pdf_file)[0]
    
    image_paths = []
    for i, img in enumerate(images):

        img_path = os.path.join(output_folder, f"{name_without_ext}_page{i+1}.png")
        img.save(img_path, "PNG")
        image_paths.append(img_path)
    
    return image_paths



# Loop through all PDFs in train_pdf folder
pdf_folder = "val_pdf"
for pdf_file in os.listdir(pdf_folder):
    if pdf_file.lower().endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, pdf_file)
        image_files = pdf_to_images(pdf_path, pdf_file)
        print(f"{pdf_file}: Converted {len(image_files)} pages to images.")