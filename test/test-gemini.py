import os
import sys
import pathlib
from google import genai
from google.genai import types

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from env_loader import load_env  # noqa: E402


load_env()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set. Add it to .env or environment variables.")

client = genai.Client(api_key=api_key)

# Đường dẫn đến tệp PDF cục bộ
local_pdf_path = "../data/pdfs/your.pdf"  # Thay bằng đường dẫn thực tế đến tệp PDF của bạn
filepath = pathlib.Path(local_pdf_path)

if not filepath.exists():
    print(f"Tệp {local_pdf_path} không tồn tại. Vui lòng kiểm tra lại đường dẫn.")
    raise SystemExit(1)

prompt = "Nói lại chi tiết về lịch Ý trang 7 tài liệu này"

response = client.models.generate_content(
    model="gemini-1.5-flash",
    contents=[
        types.Part.from_bytes(
            data=filepath.read_bytes(),
            mime_type="application/pdf",
        ),
        prompt,
    ],
)

print(response.text)
