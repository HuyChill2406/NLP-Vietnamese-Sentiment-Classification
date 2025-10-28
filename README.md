# NLP-Vietnamese-Sentiment-Classification
💡 Tổng quan

Phân loại cảm xúc tiếng Việt (Positive/Negative/Neutral) với nhiều biểu diễn (BoW/TF/TF-IDF, Doc2Vec/Word2Vec) và mô hình (Naive Bayes, Logistic Regression, Decision Tree, MLP).

📦 Dữ liệu & tiền xử lý

Nguồn: (điền mô tả + link nếu công khai).

Làm sạch: lowercase, chuẩn hoá tiếng Việt/dấu, bỏ trùng.

Chia dữ liệu: train/test = 80/20 (hoặc theo notebook của bạn).

Mất cân bằng nhãn: cân bằng nhẹ (nếu áp dụng).

🔧 Biểu diễn & Mô hình

Vectorizer: BoW (count/binary), TF, TF-IDF (1–n-gram), Doc2Vec (PV-DM/DBOW), Word2Vec.

Model: Naive Bayes, Logistic Regression, Decision Tree, MLP (PyTorch).

So khớp tham số: C/penalty cho LR, max_depth cho DT, dimension cho Doc2Vec.

📊 Đánh giá & kết quả (ví dụ mẫu)

Metric: Accuracy + Classification Report (Precision/Recall/F1 từng lớp), Confusion Matrix.

Insight:

LR + BoW/TF-IDF là baseline mạnh & ổn định.

Doc2Vec + LR tiệm cận baseline khi tăng chiều vector.

MLP cần thêm dữ liệu/regularization để vượt LR.

(Điền số liệu cụ thể từ notebook của bạn, ví dụ: “LR + Binary BoW đạt 95.27% Accuracy test”.)
