import aryn_sdk
from aryn_sdk.partition import partition_file, tables_to_pandas
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IPython.core.display import display

# Thay đổi model_id bên dưới bằng tên model phục hồi dấu tiếng Việt thực tế (nếu có)
model_id = "your-vietnamese-diacritics-model"

# Khởi tạo tokenizer và mô hình phục hồi dấu
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)


def diacritize_text(text: str) -> str:
    """
    Sử dụng mô hình phục hồi dấu để chuyển văn bản không dấu (hoặc dấu không chuẩn)
    thành văn bản có dấu tiếng Việt đầy đủ.
    """
    try:
        # Tokenize văn bản với giới hạn độ dài phù hợp
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(inputs.input_ids, max_length=512)
        diacritized_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return diacritized_text
    except Exception as e:
        print(f"Lỗi khi phục hồi dấu: {e}")
        return text


def process_pdf_normalize_cells(pdf_path: str, aryn_api_key: str, table_index: int = 1) -> pd.DataFrame:
    """
    Trích xuất bảng từ PDF sử dụng aryn_sdk, sau đó áp dụng mô hình phục hồi dấu
    cho từng ô trong bảng (DataFrame).
    """
    with open(pdf_path, "rb") as file:
        partitioned_file = partition_file(
            file,
            aryn_api_key=aryn_api_key,
            extract_table_structure=True,
            use_ocr=True  # Bật OCR để trích xuất bảng
        )

    # Lấy các bảng được trích xuất (dạng DataFrame)
    pandas_tables = tables_to_pandas(partitioned_file)
    tables = [dataframe for elt, dataframe in pandas_tables if elt['type'] == 'table']

    if len(tables) <= table_index:
        print(f"Không tìm thấy bảng ở vị trí {table_index}")
        return None

    df = tables[table_index]

    # Áp dụng hàm diacritize_text cho từng ô nếu đó là kiểu chuỗi
    normalized_df = df.applymap(lambda x: diacritize_text(x) if isinstance(x, str) else x)

    # Hiển thị kết quả bảng đã được normalize
    display(normalized_df)
    return normalized_df


if __name__ == "__main__":
    PDF_PATH = "../data/pdfs/SỔ TAY HỌC VỤ KỲ I NĂM 2023-2024.pdf"
    ARYN_API_KEY = "your-aryn-api-key"

    # Xử lý normalization cho bảng chỉ dựa trên các ô dữ liệu
    process_pdf_normalize_cells(PDF_PATH, ARYN_API_KEY, table_index=1)
