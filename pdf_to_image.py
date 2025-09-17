from pdf2image import convert_from_path
import os

def pdf_to_images(pdf_path, output_folder="pdf_images", dpi=300):
    os.makedirs(output_folder, exist_ok=True)
    images = convert_from_path(pdf_path, dpi=dpi)
    
    image_paths = []
    for i, img in enumerate(images):
        img_path = os.path.join(output_folder, f"Picking_Slip_F43987155.png")
        img.save(img_path, "PNG")
        image_paths.append(img_path)
    
    return image_paths

# Example usage
pdf_file = "ram.pdf"
image_files = pdf_to_images(pdf_file)
print(f"Converted {len(image_files)} pages to images.")