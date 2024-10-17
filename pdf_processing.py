# pdf_processing.py
import os
import fitz  # pymupdf
from PIL import Image

def pdf_page_to_base64(pdf_path: str, count: int):
    img_paths = []
    output_folder = "images"
    os.makedirs(output_folder, exist_ok=True)  # Create folder if it doesn't exist

    for page_number in range(count):
        pdf_document = fitz.open(pdf_path)
        page = pdf_document.load_page(page_number)  # input is zero-indexed
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        image_filename = os.path.join(output_folder, f"{os.path.basename(pdf_path)}_{page_number + 1}.png")
        img.save(image_filename)
        img_paths.append(image_filename)

    return img_paths  # Return the list of image paths

def count_pdf_pages(pdf_path):
    pdf_document = fitz.open(pdf_path)
    number_of_pages = pdf_document.page_count
    pdf_document.close()
    return number_of_pages