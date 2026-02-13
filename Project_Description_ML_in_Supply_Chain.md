# Norway New Car Sales Forecasting – Machine Learning Benchmark

**Multi-model time series forecasting comparison** on monthly new passenger car registrations by make in Norway.

[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-yellow)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 🎯 Mục tiêu dự án

Xây dựng, so sánh và cải tiến dần các mô hình dự báo doanh số xe hơi mới hàng tháng theo từng hãng tại Na Uy (dữ liệu từ 2007 đến nay).

Dự án đi qua các giai đoạn từ cơ bản đến nâng cao:

- Chuẩn bị dữ liệu chuỗi thời gian (sliding window)
- Sử dụng biến ngoại sinh (GDP)
- Mã hóa biến phân loại (phân khúc hãng xe, one-hot encoding)
- Phân cụm theo yếu tố mùa vụ
- Phân tích tầm quan trọng đặc trưng (feature importance)
- Dự báo đa bước (multi-step forecasting)

## 📊 Dữ liệu

- **Tệp chính**: `norway_new_car_sales_by_make.csv`
- **Nguồn gốc**: Dữ liệu công khai (Kaggle / các nguồn mở về đăng ký xe mới tại Na Uy)
- **Cấu trúc**: Dạng wide → mỗi hàng là một hãng xe, mỗi cột là tháng (YYYY-MM), giá trị là số lượng xe đăng ký mới
- **Khoảng thời gian**: Tháng 1/2007 → tháng gần nhất có dữ liệu
- **Biến ngoại sinh**: GDP hàng năm (dùng làm chỉ báo kinh tế vĩ mô đơn giản)

## 🛠 Các mô hình đã triển khai và so sánh

| Giai đoạn | Nhóm mô hình                  | Kỹ thuật / Biến thể chính                                  | Ghi chú nổi bật                     |
|-----------|-------------------------------|------------------------------------------------------------|-------------------------------------|
| 2.3       | Linear Regression             | AR(12) – tự hồi quy                                        | Baseline đơn giản                   |
| 2.4       | Decision Tree                 | max_depth=5, min_samples_split=15, so sánh MSE vs MAE      | Visualize cây quyết định            |
| 2.5       | Tuned Decision Tree           | RandomizedSearchCV (100 trials, 10-fold CV)                | Tối ưu siêu tham số                 |
| 2.6       | Random Forest                 | n_estimators=30 → 200, tuning 400 trials                   | Feature importance bar chart        |
| 2.8       | Extra Trees                   | n_estimators=200, tuning                                   | Ngẫu nhiên hơn Random Forest        |
| 2.9       | Tối ưu số lượng lag           | Thử x_len từ 6 đến 50 tháng với RF & ExtraTrees            | Tìm số tháng quá khứ tối ưu         |
| 2.10      | AdaBoost                      | Base là DecisionTree, tuning learning_rate & loss          | Phương pháp boosting                |
| 2.12      | XGBoost                       | Single-step + multi-step, early stopping, tuning 1000 trials | Mô hình mạnh nhất trong hầu hết trường hợp |
| 2.13      | Mã hóa phân loại              | Integer encoding (phân khúc) + one-hot encoding (hãng)     | Tác động riêng của từng hãng        |
| 2.14      | Phân cụm mùa vụ               | KMeans trên seasonal factors chuẩn hóa                     | Nhóm các hãng theo pattern mùa vụ   |
| 2.15      | XGBoost + feature engineering | Thống kê tổng hợp + tháng + GDP + phân khúc + cụm          | Feature selection dựa trên importance |
| 2.16      | Neural Network (MLP)          | Adam, early stopping, RandomizedSearchCV trên kiến trúc    | Baseline deep learning              |

## 📈 Kết quả chính (khoảng điển hình)

| Mô hình                        | Test MAE%   | Test RMSE%  | Bias%     | Thời gian train | Nhận xét                              |
|--------------------------------|-------------|-------------|-----------|------------------|---------------------------------------|
| Linear Regression              | 38–45%      | 60–80%      | ±5%       | <1s              | Baseline rất ổn                       |
| Decision Tree                  | 32–42%      | 55–75%      | thấp      | ~0.5s            | Dễ overfit                            |
| Tuned Tree                     | 30–38%      | 50–68%      | thấp      | 2–5s             | Cải thiện rõ                         |
| Random Forest (200 trees)      | 24–32%      | 42–58%      | 0–3%      | 5–15s            | Rất ổn định                           |
| ExtraTrees (200)               | 23–31%      | 40–56%      | thấp      | 4–12s            | Thỉnh thoảng vượt RF                  |
| XGBoost (tuned)                | **21–28%**  | **36–50%**  | 0–2%      | 3–20s            | Hiệu suất đơn mô hình tốt nhất        |
| XGBoost + rich features        | **19–26%**  | **33–47%**  | thấp      | 5–30s            | Kết quả tốt nhất trong notebook       |
| Neural Network (tuned)         | 27–38%      | 48–70%      | —         | 10–120s          | Cần tuning & dữ liệu nhiều hơn        |

*Lưu ý: kết quả thực tế thay đổi tùy thời điểm cắt train/test và random seed.*

## 📁 Cấu trúc thư mục đề xuất
Chain_Machine_Learning_Forecasting/
├── data/
│   ├── norway_new_car_sales_by_make.csv
│   └── GDP.xlsx
├── notebooks/
│   └── 01-full-forecasting-benchmark.ipynb
├── src/
│   ├── data_prep.py
│   ├── models.py
│   └── metrics.py
├── images/
│   ├── Regression_Tree.PNG
│   ├── feature_importance_xgboost.png
│   └── seasonal_clusters_heatmap.png
├── requirements.txt
├── README.md
└── .gitignore

## 🚀 Cách chạy

1. Clone repository
   ```bash
   git clone https://github.com/khaiminhdang/Chain_Machine_Learning_Forecasting.git
   cd Chain_Machine_Learning_Forecasting
2. Cài đặt môi trườngBashpip install -r requirements.txt
3. Đặt file dữ liệu vào thư mục ./data/
4. Mở và chạy python: supply_chain_ML_forecasting

