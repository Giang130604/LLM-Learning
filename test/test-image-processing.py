import pytesseract
from PIL import Image
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_images_in_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ocr_results = []
    for filename in os.listdir(input_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_dir, filename)
            try:
                img = Image.open(img_path)
                text = pytesseract.image_to_string(img, lang="vie")
                output_path = os.path.join(output_dir, f"ocr_{filename}.txt")
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(text)
                ocr_results.append((img_path, text))
            except Exception as e:
                logger.error(f"Lỗi khi xử lý {img_path}: {e}")
    return ocr_results

def main():
    input_dir = "../src/figures"
    output_dir = "./ocr_results"
    logger.info("Bắt đầu xử lý hình ảnh...")
    results = process_images_in_directory(input_dir, output_dir)
    logger.info(f"Đã xử lý {len(results)} hình ảnh.")
    for img_path, text in results:
        print(f"Hình ảnh: {img_path}")
        print(f"Văn bản OCR:\n{text}\n")

if __name__ == "__main__":
    main()