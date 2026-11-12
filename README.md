# Dự án Học Máy: Dự đoán sinh viên thôi học sớm (OULAD)

## Mô tả

Dự án sử dụng bộ dữ liệu **Open University Learning Analytics Dataset (OULAD)** để dự đoán sinh viên có nguy cơ thôi học sớm tại các mốc thời gian D=30, 60, 90 ngày.

## Thuật toán sử dụng

Chỉ sử dụng các thuật toán Machine Learning thuần (không dùng Deep Learning):

- **Phân loại**: KNN, Naive Bayes, Decision Trees, SVM (Linear), Random Forest
- **Giảm chiều**: PCA
- **Phân cụm**: K-Means (chọn k bằng quy tắc khuỷu tay/elbow method)
- **Đánh giá**: Confusion Matrix, Accuracy, Precision, Recall, F1 Score

## Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/nwee24/machine_learning_final.git
cd machine_learning_final
```

### 2. Tạo virtual environment và cài packages

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 3. Tải dataset

Dataset **KHÔNG** có trong repo để giảm dung lượng. Bạn cần tải thủ công:

**Cách 1: Tự động (chạy notebook)**
- Mở `notebook_moi.ipynb`
- Chạy cell đầu tiên → sẽ tự động tải dataset từ UCI

**Cách 2: Thủ công**
1. Tải từ: https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset
2. Giải nén vào folder `dataset/`

## Cấu trúc thư mục

```
FINALEX/
├── dataset/              # Dataset OULAD (không có trong git)
├── artifacts/            # Kết quả huấn luyện (models, plots, metrics)
│   └── cache/           # Cache features theo mốc D
├── notebook_moi.ipynb   # Notebook chính
├── app_2.py              # Streamlit app (nếu có)
├── custom_transformers.py
└── requirements.txt
```

## Chạy notebook

```bash
jupyter notebook notebook_moi.ipynb
và chạy file app_2.py bằng terminal để xem kết quả
```

hoặc mở trực tiếp trong VS Code.

## Kết quả

Tất cả kết quả được lưu trong folder `artifacts/`:
- `best_model.joblib` - Mô hình Random Forest tốt nhất
- `evaluation_results.csv` - Metrics của tất cả mô hình
- `*.png` - Các biểu đồ trực quan
- `summary.json` - Tóm tắt kết quả

## Tác giả

Võ Nhật Huy - Đồ án cuối kỳ Học Máy
