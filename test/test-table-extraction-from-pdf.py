import aryn_sdk
from IPython.core.display_functions import display
from aryn_sdk.partition import partition_file, tables_to_pandas
from google import genai
from google.genai import types
import pathlib

# Mở file PDF
file = open("../data/pdfs/SỔ TAY HỌC VỤ KỲ I NĂM 2023-2024.pdf", "rb")
aryn_api_key = "eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJpYXQiOjE3NDQxOTc0MjAsInN1YiI6eyJhY3QiOiI0NTk3NzA5OTk4NDciLCJlbWwiOiJnaWFuZ3lidm4yMDA0QGdtYWlsLmNvbSIsImdlbiI6MCwiaWQiOjEyODE2MDc2MDF9fQ.CwO8PP-svTpv1K2UJNPqtoEv_q_fS8qxuXUfkirQYrGOnPHNRUaDwdlP3oJWEtIt6o3IKAAOWrmfhyXwQANuAQ"
client = genai.Client(api_key="AIzaSyCSkPtzL-dI1fgxjCDDvBYxaDYA8z529uQ")  # Thay bằng API key thực tế của bạn
docset_id = "f5i69yhshy3v6ov3roa1oav"
# Phân tích file PDF với việc trích xuất cấu trúc bảng và sử dụng OCR
partitioned_file = partition_file(
    file,
    aryn_api_key=aryn_api_key,
    extract_table_structure=True,
    use_ocr=True,  # Đảm bảo OCR được sử dụng
    add_to_docset_id= docset_id
)

# Chuyển đổi các bảng được trích xuất sang dạng pandas DataFrame
pandas_tables = tables_to_pandas(partitioned_file)

tables = []
# Lấy ra các phần tử có kiểu là bảng
for elt, dataframe in pandas_tables:
    if elt['type'] == 'table':
        tables.append(dataframe)

# Chọn bảng thứ 2 ví dụ
supplemental_income = tables[2]

# Chuyển bảng thành chuỗi văn bản (CSV) để gửi cho Gemini
table_text = supplemental_income.to_csv(index=False)

# Xây dựng prompt cho nhiệm vụ chuẩn hóa bảng và chuyển thành list câu
prompt = "Hãy chuẩn hóa các dòng trong bảng, sửa lỗi chính tả phục hồi đầy đủ dấu và chuyển dòng thành một câu hoàn chỉnh tiếng Việt không chú thích gì thêm, chỉ liệt kê các dòng."

# Tạo Part từ text với MIME type là text/plain
table_part = types.Part.from_text(text=table_text)

# Gửi yêu cầu đến Gemini API
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[table_part, prompt]
)

# Hiển thị kết quả chuẩn hóa
print(response.text)
