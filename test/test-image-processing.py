import os
import pytesseract
from PIL import Image
import cv2
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Thư mục chứa hình ảnh
FIGURES_DIR = "../src/figures"

def extract_table_from_image(image_path: str) -> str:
    """
    Trích xuất bảng từ hình ảnh với cấu trúc (hàng, cột).
    """
    try:
        # Đọc hình ảnh bằng OpenCV
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Không thể đọc hình ảnh: {image_path}")
            return ""

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Tiền xử lý để phát hiện bảng
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Phát hiện đường kẻ ngang và dọc để xác định bảng
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

        # Kết hợp để tạo mask của bảng
        table_mask = horizontal_lines + vertical_lines
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Nếu không tìm thấy bảng, trích xuất văn bản phẳng
        if not contours:
            logger.info(f"Không phát hiện bảng trong hình ảnh {image_path}, trích xuất văn bản phẳng...")
            img_pil = Image.open(image_path)
            text = pytesseract.image_to_string(img_pil, lang="vie+eng")
            return text.strip()

        # Lấy vùng lớn nhất (giả định là bảng)
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        table_region = gray[y:y + h, x:x + w]

        # Trích xuất dữ liệu từ bảng bằng pytesseract
        data = pytesseract.image_to_data(table_region, lang="vie+eng", output_type=pytesseract.Output.DICT)

        # Tạo danh sách các ô (cells) dựa trên tọa độ
        boxes = []
        for i in range(len(data['text'])):
            # Kiểm tra độ tin cậy và đảm bảo giá trị conf là số
            conf = data['conf'][i]
            if conf == '-1':  # Bỏ qua các giá trị không phải text (conf = -1)
                continue
            try:
                conf_value = float(conf)
            except ValueError:
                logger.warning(f"Độ tin cậy không hợp lệ tại {image_path}, index {i}: {conf}")
                continue
            if conf_value > 60:  # Lọc các ô có độ tin cậy cao
                (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                text = data['text'][i].strip()
                if text:  # Chỉ thêm ô có nội dung
                    boxes.append((x, y, w, h, text))

        if not boxes:
            logger.warning(f"Không trích xuất được dữ liệu từ bảng trong {image_path}")
            return ""

        # Sắp xếp các ô theo hàng và cột
        boxes.sort(key=lambda b: (b[1], b[0]))  # Sắp xếp theo y (hàng) rồi x (cột)

        # Nhóm các ô thành hàng
        rows = []
        current_row = []
        if boxes:
            last_y = boxes[0][1]
            for box in boxes:
                x, y, w, h, text = box
                if abs(y - last_y) > h / 2:  # Nếu chênh lệch y lớn, chuyển sang hàng mới
                    if current_row:
                        rows.append(current_row)
                    current_row = []
                    last_y = y
                current_row.append(text)
            if current_row:
                rows.append(current_row)

        # Chuyển thành định dạng bảng
        if not rows:
            logger.warning(f"Không có hàng nào được trích xuất từ bảng trong {image_path}")
            return ""

        # Tạo bảng dưới dạng chuỗi văn bản có cấu trúc
        table_str = "\n".join([" | ".join(row) for row in rows])
        return table_str

    except Exception as e:
        logger.error(f"Lỗi khi trích xuất bảng từ hình ảnh {image_path}: {e}")
        return ""

def test_image_processing(figures_dir: str = FIGURES_DIR):
    """
    Test việc trích xuất bảng từ tất cả hình ảnh trong thư mục.
    """
    if not os.path.exists(figures_dir):
        logger.error(f"Thư mục không tồn tại: {figures_dir}")
        return

    image_files = [f for f in os.listdir(figures_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    if not image_files:
        logger.warning(f"Không tìm thấy hình ảnh trong thư mục: {figures_dir}")
        return

    for image_file in image_files:
        image_path = os.path.join(figures_dir, image_file)
        logger.info(f"Đang xử lý hình ảnh: {image_path}")
        table_text = extract_table_from_image(image_path)
        if table_text:
            print(f"\nHình ảnh: {image_path}")
            print(f"Nội dung trích xuất:\n{table_text}\n{'-' * 50}")
        else:
            print(f"\nHình ảnh: {image_path}")
            print("Không trích xuất được nội dung.\n" + "-" * 50)

if __name__ == "__main__":
    print("Bắt đầu test xử lý hình ảnh...")
    test_image_processing()
    print("Kết thúc test.")