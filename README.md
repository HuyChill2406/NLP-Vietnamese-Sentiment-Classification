# 📌 Midterm – Large Language Models: Tạo dataset + Phân loại cảm xúc tiếng Việt (ML/DL)

Notebook **`Midterm.ipynb`** triển khai một pipeline hoàn chỉnh cho bài toán **phân loại cảm xúc (Positive/Neutral/Negative)** trên **bình luận tiếng Việt**, bao gồm:

1. **Tạo dataset tự động bằng AI (DeepSeek Chat API)** – sinh bình luận + nhãn cảm xúc cân bằng.
2. **Tiền xử lý văn bản tiếng Việt**.
3. **Biểu diễn văn bản kiểu truyền thống (Bag of Words / TF / TF-IDF / N-grams)** + huấn luyện các mô hình ML.
4. **Biểu diễn văn bản bằng Doc2Vec (PV-DM, PV-DBOW)** và **Pretrained Word2Vec**.
5. **Huấn luyện mô hình Machine Learning & Deep Learning (PyTorch)** để so sánh.
6. **Đánh giá & trực quan hóa** (Accuracy, Precision, Recall, F1-score) + **demo dự đoán cảm xúc tương tác**.

---

## ✅ Mục tiêu

* Xây dựng tập dữ liệu cảm xúc tiếng Việt (có nhãn) một cách tự động.
* Thử nghiệm nhiều cách biểu diễn văn bản và mô hình học máy / học sâu.
* So sánh hiệu quả giữa các hướng tiếp cận.

---

## 🧠 Tổng quan nội dung Notebook

### 0) Bước tạo Dataset (DeepSeek API)

* Tạo prompt ngẫu nhiên theo nhiều chủ đề + ngẫu nhiên nhãn **Positive/Neutral/Negative** để **giảm mất cân bằng lớp**.
* Gọi API `deepseek-chat`, parse JSON trả về dạng:

  ```json
  {"comment": "...", "label": "..."}
  ```
* Lưu dữ liệu ra CSV (dấu phân cách `;`), có lọc trùng.

📌 File xuất: `comments.csv` (trong notebook phần sau đang đọc `dataset.csv` → bạn có thể **đổi tên** `comments.csv` → `dataset.csv` hoặc sửa lại dòng đọc file).

---

### 1) Tiền xử lý dữ liệu

* Kiểm tra thiếu dữ liệu, thống kê phân bố nhãn.
* Trực quan hóa bằng biểu đồ cột, wordcloud, …

---

### 2) Bag of Words + ML truyền thống

Các biến thể biểu diễn:

* CountVectorizer (BoW cơ bản)
* Binary BoW
* TF Vectorizer
* TF-IDF
* N-grams (Bigram/Trigram)

Mô hình thử nghiệm:

* Naive Bayes
* Logistic Regression
* Decision Tree
* Random Forest
  Sau đó so sánh kết quả (Accuracy/F1).

---

### 3) Doc2Vec / Pretrained Word2Vec + ML/DL

Biểu diễn:

* **Doc2Vec PV-DM**
* **Doc2Vec PV-DBOW**
* **Pretrained Word2Vec tiếng Việt** (vector size 400, huấn luyện sẵn)

Mô hình ML:

* Gaussian Naive Bayes
* Logistic Regression
* Decision Tree

Mô hình DL (PyTorch):

* MLP
* LSTM
* BiLSTM

Có phần vẽ biểu đồ/heatmap và so sánh tổng hợp ML vs DL.

---

### 4) Kết luận (trong notebook)

Notebook có ghi nhận:

* **Doc2Vec-DBOW** cho kết quả tốt nhất trong các biểu diễn thử nghiệm.
* Mô hình nổi bật:

  * **Logistic Regression + Doc2Vec-DBOW** (Test Accuracy khoảng **~0.88**)
  * **MLP + Doc2Vec-DBOW** (Test Accuracy khoảng **~0.81**)
* Decision Tree dễ overfit; BiLSTM ổn định nhưng train lâu hơn.

---

## 📁 Cấu trúc đề xuất cho repo

```
.
├── Midterm.ipynb
├── dataset.csv              # (hoặc comments.csv) dữ liệu đã gán nhãn
├── README.md
└── assets/                  # (tuỳ chọn) ảnh minh hoạ, biểu đồ
```

---

## ⚙️ Cài đặt môi trường

### Cách 1: Chạy trên Google Colab (khuyến nghị)

* Upload `Midterm.ipynb`
* (Tuỳ chọn) Runtime → Change runtime type → GPU
* Cài thư viện theo mục bên dưới

### Cách 2: Chạy local (Jupyter Notebook)

```bash
pip install -U pandas numpy scikit-learn gensim torch matplotlib seaborn wordcloud nltk tqdm requests plotly
```

📌 Nếu bạn dùng tách từ kiểu `word_tokenize(text, format="text")` như trong notebook, thường là của **underthesea**:

```bash
pip install underthesea
```

📌 NLTK (nếu cần):

```python
import nltk
nltk.download('punkt')
```

---

## 🔑 Cấu hình API (DeepSeek)

Trong notebook có phần gọi API, bạn cần thiết lập:

* `API_KEY`
* `API_URL`

Ví dụ (minh hoạ):

```python
API_KEY = "YOUR_KEY"
API_URL = "https://api.deepseek.com/chat/completions"
```

> Lưu ý: Sinh 50,000 mẫu sẽ **rất lâu** và tốn token. Trong notebook cũng có gợi ý:
>
> * 1 request trả nhiều mẫu (vd: 5)
> * chạy song song nhiều kernel

---

## ▶️ Cách chạy nhanh

1. **(Tuỳ chọn)** Chạy phần **0 – Tạo dataset** để sinh file CSV.
2. Đảm bảo file dữ liệu đúng tên:

   * Nếu bạn có `comments.csv` → đổi thành `dataset.csv` **hoặc** sửa dòng:

     ```python
     df = pd.read_csv("dataset.csv", encoding="utf-8-sig", sep=";")
     ```
3. Chạy lần lượt các section:

   * Tiền xử lý
   * BoW + ML
   * Doc2Vec/Word2Vec + ML
   * DL (MLP/LSTM/BiLSTM)
   * So sánh tổng hợp

---

## 🧪 Demo dự đoán cảm xúc (interactive)

Notebook có hàm dự đoán + vòng lặp nhập câu để test:

* Chọn phương pháp biểu diễn (DM / DBOW / Pretrained W2V)
* In ra kết quả dự đoán từ nhiều model

---

## 📌 Ghi chú quan trọng

* Dữ liệu CSV đang dùng dấu phân cách `;` và cột:

  * `Comment`
  * `Label`
* Nên giữ dataset cân bằng để tránh bias khi huấn luyện.
* Nếu bạn muốn mình viết README theo **đúng format của môn/trường** (mục tiêu, mô tả bài toán, kết quả, bảng so sánh, hướng dẫn chạy), bạn chỉ cần nói format bạn muốn (ngắn/chi tiết).
