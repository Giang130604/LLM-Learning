import os
import sys
import pathlib

import aryn_sdk
from aryn_sdk.partition import partition_file, tables_to_pandas
from google import genai
from google.genai import types

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from env_loader import load_env  # noqa: E402


load_env()
api_key = os.getenv("GEMINI_API_KEY")
aryn_api_key = os.getenv("ARYN_API_KEY")
docset_id = os.getenv("ARYN_DOCSET_ID", "")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set. Add it to .env or environment variables.")
if not aryn_api_key:
    raise ValueError("ARYN_API_KEY is not set. Add it to .env or environment variables.")

client = genai.Client(api_key=api_key)

# Mở file PDF
local_pdf_path = "../data/pdfs/your.pdf"  # Thay bằng đường dẫn thực tế
file_path = pathlib.Path(local_pdf_path)
if not file_path.exists():
    print(f"Tệp {local_pdf_path} không tồn tại. Vui lòng kiểm tra lại đường dẫn.")
    raise SystemExit(1)

file = open(local_pdf_path, "rb")

# Phân tách file PDF với việc trích xuất cấu trúc bảng và dùng OCR
partitioned_file = partition_file(
    file,
    aryn_api_key=aryn_api_key,
    extract_table_structure=True,
    use_ocr=True,
    add_to_docset_id=docset_id or None,
)

pandas_tables = tables_to_pandas(partitioned_file)
tables = [df for elt, df in pandas_tables if elt["type"] == "table"]

if not tables:
    print("Không tìm thấy bảng trong tài liệu.")
    raise SystemExit(1)

supplemental_income = tables[0]
table_text = supplemental_income.to_csv(index=False)

prompt = "Hãy chuẩn hóa các dòng trong bảng và chuyển thành câu hoàn chỉnh tiếng Việt."

response = client.models.generate_content(
    model="gemini-1.5-flash",
    contents=[
        types.Part.from_text(table_text),
        prompt,
    ],
)

print(response.text)
