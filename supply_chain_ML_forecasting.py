
# # PART 2: MACHINE LEARNING FORECASTING
# %% 
# 2.1. Data Importing and Preparation
import pandas as pd
import numpy as np
# Define a function for data importing and pivoting
def import_data():
    """Imports data from a CSV file, creates a 'Period' column, and pivots the dataframe."""
    # Import data from CSV file
    data = pd.read_csv(r"D:\norway_new_car_sales_by_make.csv")
    # Create "Period" column with format YYYY-MM (2007-01)
    data['Period'] = data['Year'].astype(str) + '-' + data['Month'].astype(str).str.zfill(2)
    # Pivot the dataframe
    df = pd.pivot_table(
        data=data,
        values='Quantity',
        index='Make',
        columns='Period',
        aggfunc='sum',
        fill_value=0)
    return df
# Run the function and display the first few rows of the result
df = import_data()
df.head()

# %%
# Data Splitting
# Define a function to split the dataset into train set and test set
# The function takes the dataframe, lengths of x and y, and number of test loops as input parameters
# Machine learning không hiểu "chuỗi thời gian".
# Vì vậy, ta cần tạo các tập dữ liệu huấn luyện và kiểm tra từ chuỗi thời gian ban đầu
# x_len = 12  # số tháng dùng làm input
# y_len = 1   # số tháng dự đoán
# test_loops = 12  # số vòng dùng làm test
# Tính số cửa sổ trượt (sliding windows). Sliding windows là cách tạo ra các tập dữ liệu huấn luyện và kiểm tra bằng cách trượt một cửa sổ qua chuỗi thời gian ban đầu
# Loops (số cửa sổ trượt (sliding windows)) = periods + 1 - x_len - y_len

def datasets(df, x_len=12, y_len=1, test_loops=12):
    """Splits the dataframe into training and testing sets based on the specified lengths and test loops."""
  
    # Get the values and shape of the dataframe
    data_values = df.values
    rows, periods = data_values.shape

    # Total number of loops (including both train and test loops)
    loops = periods + 1 - x_len - y_len

    # Create initial train set
    # Rolling window forecasting: | T1 | T2 | ... | T12 | T13 | → X = T1–T12 → Y = T13. Sau đó trượt qua: | T2 | T3 | ... | T13 | T14 |
    # Mỗi hãng đều tạo window riêng. Sau đó ghép tất cả các window của các hãng lại với nhau để tạo thành tập train chung
    # Tách X và Y để huấn luyện mô hình
    train = []
    for col in range(loops):
        train.append(data_values[:, col:col + x_len + y_len])
    train = np.vstack(train)
    X_train, Y_train = np.split(train, [-y_len], axis=1)

    # Split the initial train set into train set and test set when test_loops is specified
    # rows = số hãng xe
    # test_loops = 12
    # → 12 vòng cuối × số hãng → đưa vào test set. Tức là test trên 12 tháng gần nhất
    # Ví dụ: nếu có 10 hãng xe, thì ta sẽ lấy 120 mẫu cuối cùng làm test set
    # Còn lại làm train set
    if test_loops > 0:
        X_train, X_test = np.split(X_train, [-rows * test_loops], axis=0)
        Y_train, Y_test = np.split(Y_train, [-rows * test_loops], axis=0)
    else:
        X_test = data_values[:, -x_len:]
        Y_test = np.full((X_test.shape[0], y_len), np.nan)

    # Reformat y_train and y_test to meet scikit-learn requirements
    # Chuyển Y thành vector 1 chiều
    if y_len == 1:
        Y_train = Y_train.ravel()
        Y_test = Y_test.ravel()

    # Return the train set and test set arrays
    return X_train, Y_train, X_test, Y_test

# Run the function to split the dataset into train set and test set
X_train, Y_train, X_test, Y_test = datasets(df)

# %%
# 2.2. ML Forecasting KPIs
# Define a function to calculate forecasting accuracy KPIs
# RMSE >> MAE → có outlier lớn. Bias = trung bình sai số có dấu
# Bias > 0 → Model dự báo thấp hơn thực tế (underforecast) 
# Bias < 0 → Model dự báo cao hơn thực tế (overforecast)
# Hàm này tính toán và hiển thị các chỉ số MAE, RMSE và Bias cho cả tập huấn luyện và tập kiểm tra

def kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name=''):
    """Calculate and display MAE, RMSE, and Bias for train and test sets."""

    # Initialize dataframe to store the results
    df = pd.DataFrame(columns=['MAE', 'RMSE', 'Bias'], index=['Train', 'Test'])
    df.index.name = name

    # Calculate metrics for the train set
    df.loc['Train', 'MAE'] = 100 * np.mean(abs(Y_train - Y_train_pred)) / np.mean(Y_train)
    df.loc['Train', 'RMSE'] = 100 * np.sqrt(np.mean((Y_train - Y_train_pred)**2)) / np.mean(Y_train)
    df.loc['Train', 'Bias'] = 100 * np.mean((Y_train - Y_train_pred)) / np.mean(Y_train)

    # Calculate metrics for the test set
    df.loc['Test', 'MAE'] = 100 * np.mean(abs(Y_test - Y_test_pred)) / np.mean(Y_test)
    df.loc['Test', 'RMSE'] = 100 * np.sqrt(np.mean((Y_test - Y_test_pred)**2)) / np.mean(Y_test)
    df.loc['Test', 'Bias'] = 100 * np.mean((Y_test - Y_test_pred)) / np.mean(Y_test)

    # Format the dataframe for better presentation
    df = df.astype(float).round(1)

    # Print the results
    print(df)

# %%
# 2.3. Linear Regression
from sklearn.linear_model import LinearRegression

# Setup model and fit train set
reg = LinearRegression()
reg.fit(X_train, Y_train)

# Forecast and return forecasting accuracy KPIs
# 12 biến 𝑥 x = 12 tháng trước. Y = tháng tiếp theo ==> Nhu cầu tháng tiếp theo là tổ hợp tuyến tính của 12 tháng trước
# Autoregressive model với độ trễ 12 tháng (AR(12))
Y_train_pred = reg.predict(X_train)
Y_test_pred = reg.predict(X_test)

kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name='Regression')
# %%
# Now use the model to forecast
# Forecast for the future (with test_loops = 0)
# test_loops = 0 → không có test set (Trước đó ta giữ lại 12 tháng cuối làm test)
# Lúc này, ta không có test set nữa vì ta muốn dự báo cho những tháng tiếp theo sau tháng cuối cùng trong dữ liệu ban đầu
# Ta sẽ dùng toàn bộ dữ liệu để train mô hình, sau đó dự báo cho những tháng tiếp theo
# 
X_train_reg, Y_train_reg, X_test_reg, Y_test_reg = datasets(df, x_len=12, y_len=1, test_loops=0)

reg = LinearRegression()
reg.fit(X_train_reg, Y_train_reg)

forecast = pd.DataFrame(
    data=reg.predict(X_test_reg),
    index=df.index,
    columns=['Forecasting result'])
print(forecast.head())

# %%
# 2.4. Decision Tree
# Run Decision Tree Regressor model
from sklearn.tree import DecisionTreeRegressor

# Setup model and fit train set
# Cây tối đa 5 tầng → Giới hạn độ phức tạp → Giảm overfitting
# Một node chỉ được chia tiếp nếu có ít nhất 15 mẫu → Tránh chia quá nhỏ → Tránh noise learning
# Mỗi lá phải có ít nhất 5 quan sát → Dự báo tại mỗi lá = trung bình của ít nhất 5 điểm → Làm dự báo ổn định hơn
# Tree học bằng cách: Chọn 1 feature (ví dụ Y_{t-1}) 
# Chọn 1 ngưỡng (ví dụ 120) & Chia data thành 2 nhóm: Nhóm ≤ 120 Nhóm > 120 
# Tối thiểu hóa MSE sau khi chia & Lặp lại cho từng node
# Quá trình dừng khi đạt max_depth hoặc min_samples_split hoặc min_samples_leaf
# Dự báo tại mỗi lá = trung bình của các điểm trong lá đó
# Ví dụ: Lá có 7 điểm với giá trị Y là {100, 110, 120, 130, 140, 150, 160} → Dự báo tại lá này = (100+110+120+130+140+150+160)/7 = 130
# Cây càng sâu, càng nhiều lá → Dự báo càng chi tiết → Nhưng dễ overfitting
# Decision Tree là mô hình phi tuyến + no formula + easy to be overfit + capture natural interaction → Mô hình hóa các quan hệ phức tạp hơn Linear Regression

tree = DecisionTreeRegressor(max_depth=5, min_samples_split=15, min_samples_leaf=5)
tree.fit(X_train, Y_train)

# Forecast and return forecasting accuracy KPIs
Y_train_pred = tree.predict(X_train)
Y_test_pred = tree.predict(X_test)

kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name='Tree')

# %%
# Visualize the tree

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Create figure and axis for the tree visualization
fig = plt.figure(figsize=(15, 6), dpi=300)
ax = fig.add_subplot(111)

# Visualize the tree and save as an image
plot_tree(tree, fontsize=3, feature_names=[f'M{x-12}' for x in range(12)],
          rounded=True, filled=True, ax=ax)
fig.savefig('Regression_Tree.PNG')

# %%
# Forecast accuracy and time comparison between criterion MSE and MAE
# Nghĩa là so sánh giữa việc sử dụng MSE (squared_error) và MAE (absolute_error) làm tiêu chí chia node trong cây quyết định
# MSE nhạy cảm với outlier lớn, trong khi MAE ít nhạy cảm hơn với outlier
# Việc so sánh này giúp hiểu rõ hơn về ảnh hưởng của tiêu chí chia node
# Đồng thời, đo thời gian huấn luyện để đánh giá hiệu suất của từng tiêu chí

import time
# Dictionary to store results
results = []

# Loop through different criteria
# Ghi lại thời điểm bắt đầu
for criterion in ['squared_error', 'absolute_error']:
    start_time = time.time()

    # Initialize and fit the model
    # Thiết lập và huấn luyện mô hình với tiêu chí hiện tại
    tree = DecisionTreeRegressor(
        max_depth=5, min_samples_split=15, min_samples_leaf=5, criterion=criterion)
    tree.fit(X_train, Y_train)

    # Predict and evaluate KPIs
    Y_train_pred = tree.predict(X_train)
    Y_test_pred = tree.predict(X_test)
    kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name=f'Tree {criterion}')
    print()

    # Record training time
    # → Thời gian = thời điểm hiện tại - lúc bắt đầu
    training_time = time.time() - start_time
    results.append([criterion, training_time])

# Convert results to DataFrame for easier comparison
# Lưu training time và convert sang DataFrame để dễ so sánh
results_df = pd.DataFrame(results, columns=['Criterion', 'Training Time (seconds)'])
print(results_df)

# %%
# 2.5. Parameter Optimization (Decision Tree examples)
# Use Randomized Search with Cross-Validation to optimize Decision Tree parameters
# Tối ưu hóa tham số của Decision Tree bằng cách sử dụng Randomized Search kết hợp với Cross-Validation

from sklearn.model_selection import RandomizedSearchCV
# Parameter grid
# max_depth 5 → 10: cây từ vừa đến sâu. None: không giới hạn độ sâu
# min_samples_split 5 → 20: mỗi node phải có ít nhất 5-20 mẫu mới được chia tiếp
# min_samples_leaf 2 → 20: mỗi lá phải có ít nhất 2-20 mẫu

max_depth = list(range(5, 11)) + [None]
min_samples_split = range(5, 20)
min_samples_leaf = range(2, 20)
# Sau đó gom lại thành dictionary param_dist Đây là không gian để Random Search thử nghiệm
# Mỗi lần thử nghiệm, Random Search sẽ chọn ngẫu nhiên một tổ hợp các tham số từ param_dist để huấn luyện mô hình và đánh giá hiệu suất bằng Cross-Validation
param_dist = {
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf}

# Setup model
# Initialize Decision Tree Regressor (Khởi tạo model cơ bản), sau đó RandomizedSearchCV sẽ tìm tham số tốt nhất dựa trên không gian param_dist
tree = DecisionTreeRegressor()

# Apply K-Fold Cross-Validation & Random Search with MAE scoring
# RandomizedSearchCV sẽ thử nghiệm 100 tổ hợp tham số khác nhau (n_iter=100) (Không thử hết toàn bộ tổ hợp. Vì số tổ hợp quá lớn)
# Sử dụng 10-fold Cross-Validation (cv=10) để đánh giá hiệu suất của mỗi tổ hợp tham số
# Sử dụng MAE làm tiêu chí đánh giá (scoring='neg_mean_absolute_error')
# 10-fold cross validation | Quy trình: Chia dữ liệu thành 10 phần & Train trên 9 phần. Test trên 1 phần Lặp lại 10 lần 
# Lấy trung bình MAE → Giảm rủi ro phụ thuộc 1 cách chia train/test
# scoring='neg_mean_absolute_error' => Trả về giá trị âm của MAE vì sklearn tối ưu hóa hàm điểm số sao cho càng lớn càng tốt 
# Trong khi MAE càng nhỏ càng tốt → Đổi dấu để phù hợp với yêu cầu của sklearn
# verbose=1 → Hiển thị tiến trình của quá trình tìm kiếm
tree_cv = RandomizedSearchCV(
    estimator=tree, param_distributions=param_dist, n_iter=100,
    scoring='neg_mean_absolute_error', n_jobs=-1, cv=10, verbose=1)
tree_cv.fit(X_train, Y_train)

# Output the best parameters and score
print('Tuned Regression Tree Parameters:', tree_cv.best_params_)
print('Best Cross-Validation MAE:', -tree_cv.best_score_)  # Negate to get positive MAE

# %%
# Use the tuned model with optimized parameters to forecast and return forecasting accuracy KPIs
# tree_cv không còn là cây mặc định nữa mà sau khi .fit(), nó: Đã chạy 100 random cấu hình + Đã chọn bộ tham số tốt nhất + Tự động refit lại model trên toàn bộ X_train với best params 
# Nên: tree_cv.predict() = dùng cây tối ưu rồi
# Lấy mô hình với tham số tối ưu từ RandomizedSearchCV để dự báo và đánh giá KPIs
y_train_pred = tree_cv.predict(X_train)
y_test_pred = tree_cv.predict(X_test)

kpi_ML(Y_train, y_train_pred, Y_test, y_test_pred, name='Tree Tuned')
print()

# Check the detail K-Fold Cross-Validation & Random Search result
cv_result = pd.DataFrame(tree_cv.cv_results_)
print(cv_result.head())

# %%
# 2.6. Random Forest
# Train Random Forest với tham số cố định
# Random Forest Là trung bình dự báo của nhiều Decision Tree
# Mỗi tree học trên 1 mẫu con (bootstrap sample) của tập huấn luyện (Mỗi cây được train trên mẫu random có lặp (sampling with replacement để tạo sự khác biệt giữa các cây và giảm variance))
# Mỗi lần chia node, chỉ chọn 1 tập con của các biến: Mỗi cây chỉ dùng 95% dữ liệu train → tăng randomness → giảm overfitting
# Mỗi node chỉ được chọn 11 feature để split → làm các cây khác nhau → giảm correlation giữa cây
# Một leaf phải có ít nhất 18 điểm & max_depth=7 Giới hạn độ sâu cây → ngăn overfitting
from sklearn.ensemble import RandomForestRegressor
# Setup model and fit train set
forest = RandomForestRegressor(
    bootstrap=True,
    max_samples=0.95,
    max_features=11,
    min_samples_leaf=18,
    max_depth=7)
forest.fit(X_train, Y_train)

# Forecast and return forecasting accuracy KPIs
Y_train_pred = forest.predict(X_train)
Y_test_pred = forest.predict(X_test)
kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name='Forest')

# %%
# Parameter optimization with n_estimators=30
# Random Forest có nhiều tham số (hyperparameter) hơn Decision Tree để tối ưu hóa, bao gồm: max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap, max_samples ảnh hưởng đến: Bias Variance Stability Speed
# Tuning Random Forest ==> RandomizedSearchCV thử 400 tổ hợp ngẫu nhiên: n_iter=400 và cv=6 ==> 400 cấu hình Mỗi cấu hình chạy 6 folds = 2400 lần train model
# Parameter grid
max_depth = list(range(5, 11)) + [None]
min_samples_split = range(5, 20)
min_samples_leaf = range(2, 15)
max_features = range(3, 8)
bootstrap = [True]
max_samples = [.7, .8, .9, .95, 1]
param_dist = {
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'max_features': max_features,
    'bootstrap': bootstrap,
    'max_samples': max_samples}

# Apply K-Fold Cross-Validation & Random Search with MAE scoring to the model
forest = RandomForestRegressor(n_jobs=1, n_estimators=30)
forest_cv = RandomizedSearchCV(
    estimator=forest,
    param_distributions=param_dist,
    cv=6,
    n_jobs=-1,
    verbose=2,
    n_iter=400,
    scoring='neg_mean_absolute_error')
forest_cv.fit(X_train, Y_train)
print('Tuned Forest Parameters:', forest_cv.best_params_)

# Use the tuned model with optimized parameters to forecast and return forecasting accuracy KPIs
Y_train_pred = forest_cv.predict(X_train)
Y_test_pred = forest_cv.predict(X_test)
kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name='Forest optimized')

# %%
# Parameter optimization with n_estimators=200
# Use the tuned model with optimized parameters and n_estimators = 200 to forecast and return forecasting accuracy KPIs
# Tăng n_estimators từ 30 lên 200 để cải thiện độ ổn định và độ chính xác của mô hình
# Số lượng cây càng nhiều, dự báo càng ổn định và chính xác hơn, nhưng thời gian huấn luyện cũng tăng lên
# Dùng tham số tối ưu từ bước trước (Giữ toàn bộ cấu hình tối ưu đã tìm được), chỉ thay n_estimators = 200
# Random Forest không overfit khi tăng số cây

forest = RandomForestRegressor(n_estimators=200, n_jobs=-1, **forest_cv.best_params_)
forest = forest.fit(X_train, Y_train)
Y_train_pred = forest.predict(X_train)
Y_test_pred = forest.predict(X_test)
kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name='Forest n_estimators = 200')

# %%
# 2.7. Feature Importance (Random Forest examples)
# Xem mô hình Random Forest “đang dựa vào tháng nào nhiều nhất” để dự báo
# --> Tháng nào trong quá khứ ảnh hưởng mạnh nhất đến dự báo tháng tiếp theo
# Random Forest tính importance dựa trên: Tổng mức giảm impurity (MSE) do mỗi feature tạo ra
# Khi một feature được dùng để split: Nếu nó làm giảm MSE nhiều Và được dùng nhiều lần → importance cao
# Number of train features
cols = X_train.shape[1]

# Get the feature list
features = [f'M-{cols - col}' for col in range(cols)]

# Create the feature importance dataframe
feature_importance = pd.DataFrame(data=forest.feature_importances_.reshape(-1, 1),
                                  index=features,
                                  columns=['Forest'])

# Visualize the feature importance chart
feature_importance.plot(kind='bar')

# %%
# 2.8. Extremely Randomized Trees/Extra Trees
# Random Forest → Extra Trees (Extremely Randomized Trees)
# Extra trees tương tự Random Forest, nhưng có thêm độ ngẫu nhiên:
# Chia node bằng cách chọn ngưỡng split ngẫu nhiên thay vì tìm ngưỡng tối ưu
# Giúp giảm variance hơn nữa → Mô hình ổn định hơn nữa
# Tuy nhiên, do tăng độ ngẫu nhiên nên bias có thể tăng nhẹ

# Giải thích thêm về Extra Trees
# Random Forest: Chọn subset feature ngẫu nhiên Nhưng vẫn tìm best split tối ưu (giảm MSE nhiều nhất)
# Extra Trees: Chọn feature ngẫu nhiên Và còn chọn ngưỡng split ngẫu nhiên luôn Không tìm best threshold

from sklearn.ensemble import ExtraTreesRegressor
# Setup the model
# Extra Trees vẫn là ensemble của nhiều decision trees nhưng có thêm độ ngẫu nhiên trong việc chọn ngưỡng split
ETR = ExtraTreesRegressor(n_jobs=-1, n_estimators=200, min_samples_split=15,
                          min_samples_leaf=4, max_samples=0.95, max_features=4,
                          max_depth=8, bootstrap=True)

# Fit train set to the model
ETR.fit(X_train, Y_train)

# Use the model to predict train and test sets
Y_train_pred = ETR.predict(X_train)
Y_test_pred = ETR.predict(X_test)

# Return forecasting accuracy KPIs
kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name='ETR')

# %%
# Parameter optimization with n_estimators=30 - Tuning với n_estimators=30
# Tối ưu hóa tham số của Extra Trees bằng cách sử dụng Randomized Search kết hợp với Cross-Validation
# # Parameter grid
max_depth = list(range(6, 13)) + [None]
min_samples_split = range(7, 16)
min_samples_leaf = range(2, 13)
max_features = range(5, 13)
bootstrap = [True]
max_samples = [.7, .8, .9, .95, 1]

param_dist = {'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf,
              'max_features': max_features,
              'bootstrap': bootstrap,
              'max_samples': max_samples}

ETR = ExtraTreesRegressor(n_jobs=1, n_estimators=30)
ETR_cv = RandomizedSearchCV(ETR, param_dist, cv=5, verbose=2, n_jobs=-1,
                            n_iter=400, scoring='neg_mean_absolute_error')
ETR_cv.fit(X_train, Y_train)

print('Tuned Forest Parameters:', ETR_cv.best_params_)

# Use the tuned model to predict train and test sets
Y_train_pred = ETR_cv.predict(X_train)
Y_test_pred = ETR_cv.predict(X_test)
kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name='ETR optimized')

# %%
# Parameter optimization with n_estimators=200 - Tuning với n_estimators=200
# Use the tuned model with optimized parameters and n_estimators = 200 to forecast and return forecasting accuracy KPIs
# Run the tuned model with 200 trees
ETR = ExtraTreesRegressor(n_estimators=200, n_jobs=-1, **ETR_cv.best_params_).fit(X_train, Y_train)
Y_train_pred = ETR.predict(X_train)
Y_test_pred = ETR.predict(X_test)
kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name='ETR x200')

# %%
# 2.9. Feature Optimization #1 (Random Forest and Extremely Randomized Trees examples)
# Feature optimization: Chọn ra các biến quan trọng nhất để huấn luyện mô hình
# Giúp giảm độ phức tạp mô hình, tăng tốc độ huấn luyện, và có thể cải thiện hiệu suất nếu loại bỏ được các biến không quan trọng

# Determine the optimal number of feature using train set
# hay là số tháng dùng làm input (Dùng bao nhiêu tháng quá khứ (bao nhiêu lag) là tối ưu?)
# Tức là đang tối ưu feature space, không phải hyperparameter nữa

# Get the dataframe
df = import_data()

# Define RandomForestRegressor parameters
# Test với: Random Forest Extra Trees & giữ hyperparameter cố định (đã tối ưu từ trước)
forest_features = {
    "n_jobs": -1,
    "n_estimators": 200,
    "min_samples_split": 15,
    "min_samples_leaf": 4,
    "max_samples": 0.95,
    "max_features": 0.3,
    "max_depth": 8,
    "bootstrap": True
}
forest = RandomForestRegressor(**forest_features)

# Define ExtraTreesRegressor parameters
etr_features = {
    "n_jobs": -1,
    "n_estimators": 200,
    "min_samples_split": 14,
    "min_samples_leaf": 2,
    "max_samples": 0.9,
    "max_features": 1.0,
    "max_depth": 12,
    "bootstrap": True
}
etr = ExtraTreesRegressor(**etr_features)

# List of models
models = [("Forest", forest), ("ETR", etr)]

# Create function to return MAE
# Sai số tương đối so với quy mô dữ liệu
# MAE% = Mean Absolute Error / Mean Actual Value
def model_mae(model, X, y):
    y_pred = model.predict(X)
    mae = np.mean(np.abs(y - y_pred)) / np.mean(y)
    return mae

# Define range for months (from 6 to 50 with a 2-month gap)
# Số tháng dùng làm input (feature) từ 6 đến 50, bước nhảy 2 tháng
# Tức là sẽ thử với 6, 8, 10, ..., 48
n_months = range(6, 50, 2)

# Empty list to store the results
results = []

# Train and test models to find the optimal number of features
# Mỗi lần: Tạo bộ feature mới Số cột X thay đổi Mô hình phải học lại
# Lưu lại MAE% cho train và test set
for x_len in n_months:
    X_train, Y_train, X_test, Y_test = datasets(df, x_len=x_len)

    for name, model in models:
        model.fit(X_train, Y_train)
        mae_train = model_mae(model, X_train, Y_train)
        mae_test = model_mae(model, X_test, Y_test)

        results.append([f"{name} Train", mae_train, x_len])
        results.append([f"{name} Test", mae_test, x_len])

# Format results into a DataFrame for visualization
data = pd.DataFrame(results, columns=["Model", "MAE%", "Number of Months"])
data = data.set_index(["Number of Months", "Model"]).stack().unstack("Model")
data.index = data.index.droplevel(level=1)
data.index.name = "Number of Months"

# Visualize the results
data.plot(color=["orange"] * 2 + ["black"] * 2, style=["-", "--"] * 2)
plt.xlabel("Number of Months")
plt.ylabel("MAE%")
plt.title("Model Performance Across Different Time Periods")
plt.show()

# Print the optimal number of features
print(data.idxmin())

# %%
# 2.10. Adaptive Boosting/AdaBoost
# Khác với Random Forest và Extra Trees là xây dựng nhiều cây độc lập rồi lấy trung bình dự báo
# AdaBoost xây dựng các cây theo chuỗi, mỗi cây sau tập trung sửa lỗi của cây trước 
# Quá trình này giúp mô hình học từ các lỗi trước đó và cải thiện độ chính xác dự báo
# Mỗi cây trong chuỗi được huấn luyện trên dữ liệu đã được điều chỉnh trọng số để tập trung vào các điểm dữ liệu mà cây trước đó dự báo sai
# Cuối cùng, dự báo của tất cả các cây được kết hợp lại để tạo thành dự báo cuối cùng
# AdaBoost thường sử dụng các cây nông (shallow trees) làm weak learners để tránh overfitting và giữ mô hình đơn giản
# Việc kết hợp nhiều weak learners giúp mô hình tổng thể mạnh mẽ hơn và có khả năng tổng quát hóa tốt hơn trên dữ liệu mới
# AdaBoost phù hợp cho các bài toán dự báo phức tạp, nơi mà các mô hình đơn lẻ có thể không đủ mạnh để nắm bắt các mẫu trong dữ liệu
# Trước đây dùng: Bagging (song song, giảm variance) --> Bây giờ dùng: Boosting (tuần tự, giảm bias)
# Boosting: Kết hợp nhiều mô hình yếu (weak learners) thành mô hình mạnh (strong learner)

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

# DecisionTreeRegressor(max_depth=8) --> Đây là base learner
# n_estimators=100 Số lượng cây được build tuần tự
# learning_rate=0.25 Tốc độ học (learning rate) kiểm soát mức độ ảnh hưởng của mỗi cây mới được thêm vào mô hình
# Nhỏ → học chậm → ổn định hơn
# Lớn → học nhanh → dễ overfit
ada = AdaBoostRegressor(DecisionTreeRegressor(max_depth=8), n_estimators=100, learning_rate=0.25, loss='square')
ada = ada.fit(X_train, Y_train)

Y_train_pred = ada.predict(X_train)
Y_test_pred = ada.predict(X_test)
kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name='AdaBoost')

# %%
# Parameter optimization
# Parameter grid
n_estimators = [100]
learning_rate = [0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
loss = ['square', 'exponential', 'linear']

param_dist = {# 'n_estimators': n_estimators,  # Uncomment decide to test this parameter
              'learning_rate': learning_rate,
              'loss': loss}

from sklearn.model_selection import RandomizedSearchCV

# List to store results
results = []

# Loop over different max_depth values
for max_depth in range(2, 18, 2):
    ada = AdaBoostRegressor(DecisionTreeRegressor(max_depth=max_depth))
    ada_cv = RandomizedSearchCV(ada, param_dist, n_jobs=-1, cv=6, n_iter=20, scoring='neg_mean_absolute_error')
    ada_cv.fit(X_train, Y_train)
    print(f'Tuned AdaBoost Parameters for max_depth={max_depth}:', ada_cv.best_params_)
    print('Result:', ada_cv.best_score_)

    # Store the results
    results.append([ada_cv.best_score_, ada_cv.best_params_, max_depth])

# Convert results to DataFrame for easy visualization
results_df = pd.DataFrame(results, columns=['Best Score', 'Best Parameters', 'Max Depth'])
print(results_df)

# Convert the results to DataFrame
results = pd.DataFrame(data=results, columns=['Score', 'Best Params', 'Max Depth'])

# Find the index of the maximum score
# best_score_ = neg_mean_absolute_error → Giá trị càng gần 0 càng tốt
optimal = results['Score'].idxmax()

# Print the row corresponding to the optimal score
print(results.iloc[optimal])

# %%
# Test the optimized model with loss function linear to check the result
ada = AdaBoostRegressor(
    DecisionTreeRegressor(max_depth=8),
    n_estimators=100,
    learning_rate=0.005,
    loss="linear",)

ada.fit(X_train, Y_train)
y_train_pred = ada.predict(X_train)
y_test_pred = ada.predict(X_test)
kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name="AdaBoost optimized")

# %%
# Use AdaBoost with MultiOutputRegressor to forecast multiple output values
from sklearn.multioutput import MultiOutputRegressor

multi = MultiOutputRegressor(ada, n_jobs=-1)
X_train, Y_train, X_test, Y_test = datasets(df, x_len=12, y_len=6, test_loops=12)
multi.fit(X_train, Y_train)

# %%
# 2.11. Demand Drivers and Leading Indicators
# Import dataset
df = import_data()
GDP = pd.read_excel("C:\\Users\\DELL\\Downloads\\GDP.xlsx").set_index('Year')
dates = pd.to_datetime(df.columns,format='%Y-%m').year
X_GDP = [GDP.loc[date,'GDP'] for date in dates]

# Define a function to split the dataset into train set and test set (with exogenous data input)
# Ý tưởng: Trước giờ chỉ dùng: pure time series (autoregressive) ==> Dự báo dựa trên chính dữ liệu quá khứ của nó
# Bây giờ, ta sẽ thêm vào các yếu tố bên ngoài (exogenous variables)
# Dự báo demand không chỉ dựa vào demand quá khứ mà còn dựa vào yếu tố kinh tế vĩ mô (GDP)

def datasets_exo(df, X_exo, x_len=12, y_len=1, test_loops=12):

  # Get the value and shape of the dataframe
  D = df.values
  rows, periods = D.shape

  # Reshape X_exo to a row then repeat that row multiple times to reach the amount of rows in the dataframe
  X_exo = np.repeat(np.reshape(X_exo,[1,-1]), rows, axis=0)

  # Create an array X_months that contains the last month of each period then repeat it multiple times to reach the amount of rows in the dataframe
  X_months = np.repeat(np.reshape([int(col[-2:]) for col in df.columns], [1,-1]), rows, axis=0)

  # Total number of loops, including train and test in the dataset
  loops = periods + 1 - x_len - y_len

  # Create train set and test set
  # For each column in total loop, take all data from that column to the column at the end of a loop (13 months)
  # m = X_months[:,col+x_len] -- m → tháng hiện tại
  # exo = X_exo[:,col:col+x_len] -- exo → GDP của x_len tháng trước
  # d = D[:,col:col+x_len+y_len] -- d → lag demand
  # XGBoost là Gradient Boosting nâng cấp
  # AdaBoost: Update weight sample + Không regularization mạnh + Ít tối ưu hóa
  # XGBoost: Fit residual + Regularization mạnh + Nhiều tối ưu hóa hơn

  train = []
  for col in range(loops):
    m = X_months[:,col+x_len].reshape(-1,1) #month
    exo = X_exo[:,col:col+x_len] #exogenous data
    d = D[:,col:col+x_len+y_len]
    train.append(np.hstack([m, exo, d]))
  train = np.vstack(train)
  X_train, Y_train = np.split(train,[-y_len],axis=1)

  # If test_loops is required, split the X_train, Y_train above to train set and test set
  # Else, X_test is used to generate the future forecast and Y_test contains dummy values
  if test_loops > 0:
    X_train, X_test = np.split(X_train, [-rows*test_loops], axis = 0)
    Y_train, Y_test = np.split(Y_train, [-rows*test_loops], axis = 0)
  else:
    X_test = np.hstack([m[:,-1].reshape(-1,1),X_exo[:,-x_len:],D[:,-x_len:]])
    Y_test = np.full((X_test.shape[0], y_len), np.nan)

  # Reformat Y_train and Y_test to meet scikit-learn requirement
  if y_len == 1:
    Y_train = Y_train.ravel()
    Y_test = Y_test.ravel()

  # Return test set and train set
  return X_train, Y_train, X_test, Y_test

# %%
# 2.12. Extreme Gradient Boosting/XGBoost
# 2.12.1. Run the model
from xgboost.sklearn import XGBRegressor
XGB = XGBRegressor(
    n_jobs=-1,
    max_depth=10,
    n_estimators=100,
    learning_rate=0.2)
XGB = XGB.fit(X_train, Y_train)

# 2.12.2. Feature Importance

# Gain = tổng mức giảm loss do feature đó gây ra
# Khác Random Forest: RF importance = giảm MSE trung bình còn XGB importance = tổng gain qua boosting rounds
# Gain phản ánh mức độ quan trọng thực sự của feature trong việc cải thiện hiệu suất mô hình
# Gain được tính toán bằng cách tổng hợp mức giảm loss (ví dụ: MSE) mà mỗi feature đóng góp trong quá trình xây dựng các cây quyết định trong mô hình XGBoost
# Feature với gain cao → quan trọng hơn → model dựa vào nhiều hơn để dự báo

import xgboost as xgb
XGB.get_booster().feature_names = [f'M{x-12}' for x in range(12)]
xgb.plot_importance(XGB, importance_type='total_gain', show_values=False)

# %%

# 2.12.3. Use XGBoost with MultiOutputRegressor to forecast multiple output values

# Nâng y_len = 6 == > Dự báo 6 tháng tiếp theo cùng lúc
from sklearn.multioutput import MultiOutputRegressor

# Training and testing
# Multi-step forecasting: Direct strategy (Mỗi horizon có model riêng) and Recursive strategy (Dự báo t+1 rồi dùng nó dự báo t+2)
X_train, Y_train, X_test, Y_test = datasets(
    df, x_len=12, y_len=6, test_loops=12)
XGB = XGBRegressor(
    n_jobs=1,
    max_depth=10,
    n_estimators=100,
    learning_rate=0.2)
multi = MultiOutputRegressor(XGB, n_jobs=-1)
multi.fit(X_train, Y_train)

# Future forecast
# Nghĩa là: Train trên toàn bộ lịch sử Tạo X_test là 12 tháng cuối Dự báo 6 tháng tương lai
# 12 tháng cuối không dùng làm test set nữa mà dùng để tạo input X_test cho việc dự báo tương lai
X_train, Y_train, X_test, Y_test = datasets(
    df, x_len=12, y_len=6, test_loops=0)
XGB = XGBRegressor(
    n_jobs=1,
    max_depth=10,
    n_estimators=100,
    learning_rate=0.2)
multi = MultiOutputRegressor(XGB, n_jobs=-1)
multi.fit(X_train, Y_train)
forecast = pd.DataFrame(data=multi.predict(X_test), index=df.index)
forecast.head()

# %%
# 2.12.4. Early Stopping when reaching the minimal loss function value of evaluation set
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.15)

XGB = XGBRegressor(n_jobs=-1,
                   max_depth=10,
                   n_estimators=1000,
                   learning_rate=0.01,
                   objective='reg:absoluteerror',
                   early_stopping_rounds=100)

# Only use validation set for early stoppping evaluation
XGB = XGB.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
print(f'Using validation set for evaluation')
print(f'Best iteration: {XGB.get_booster().best_iteration}')
print(f'Best score: {XGB.get_booster().best_score}')
print()

# Use both train set and validation set for early stoppping evaluation
XGB = XGB.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_val, y_val)], verbose=False)
print(f'Using train set and validation set for evaluation')
print(f'Best iteration: {XGB.get_booster().best_iteration}')
print(f'Best score: {XGB.get_booster().best_score}')
print()

# Use holdout set for early stoppping evaluation
X_train, Y_train, X_holdout, Y_holdout, X_test, Y_test = datasets_holdout(
    df, x_len=12, y_len = 1, test_loops = 12, holdout_loops = 12
)

XGB = XGB.fit(X_train, Y_train, eval_set=[(X_holdout, Y_holdout)], verbose=False)
print(f'Using holdout set for evaluation')
print(f'Best iteration: {XGB.get_booster().best_iteration}')
print(f'Best score: {XGB.get_booster().best_score}')
print()

# %%
# 2.12.5. (PENDING) Early Stopping for XGBoost with MultiOutputRegressor --> Cannot use eval_set with MultiOutputRegressor
from sklearn.multioutput import MultiOutputRegressor
X_train, Y_train, X_test, Y_test = datasets(
    df, x_len=12, y_len=6, test_loops=0)

x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.15)

XGB = XGBRegressor(
    n_jobs=1,
    max_depth=10,
    n_estimators=100,
    learning_rate=0.2,
    objective='reg:absoluteerror',
    early_stopping_rounds=25,)
multi = MultiOutputRegressor(XGB, n_jobs=-1)
multi.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)

# %%
# 2.12.6. Parameter optimization
# Train, test, and validation sets
X_train, Y_train, X_test, Y_test = datasets(
    df, x_len=12, y_len=6, test_loops=12)
x_train, x_val, y_train, y_val = train_test_split(
    X_train, Y_train, test_size=0.15)

# Parameter grid
params = {
    'max_depth': [5, 6, 7, 8, 10, 11],
    'learning_rate': [0.005, 0.01, 0.025, 0.05, 0.1, 0.15],
    'colsample_bynode': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bylevel': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'subsample': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    'min_child_weight': [5, 10, 15, 20, 25],
    'reg_alpha': [1, 5, 10, 20, 50],
    'reg_lambda': [0.01, 0.05, 0.1, 0.5, 1],
    'n_estimators': [1000],}

# Set up model
XGB = XGBRegressor(
    n_jobs=1, early_stopping_rounds=25, objective='reg:absoluteerror')

# Random Search
XGB_cv = RandomizedSearchCV(
    XGB,
    params,
    cv=5,
    n_jobs=-1,
    verbose=1,
    n_iter=1000,
    scoring='neg_mean_absolute_error',)
XGB_cv.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
print('Tuned XGBoost Parameters:', XGB_cv.best_params_)

# %%
# Train the final model with optimized parameters
best_params = XGB_cv.best_params_

XGB = XGBRegressor(
    n_jobs=-1,
    early_stopping_rounds=25,
    objective='reg:absoluteerror',
    **best_params)

XGB.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)

# Print best iteration and score
print(f'Best iteration: {XGB.get_booster().best_iteration}')
print(f'Best score: {XGB.get_booster().best_score}')

# Make predictions and evaluate performance
Y_train_pred = XGB.predict(X_train)
Y_test_pred = XGB.predict(X_test)
kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name='XGBoost')

# %%
# 2.13. Categorical Features
# 2.13.1. Integer Encoding
# Define the segment for each car brand
luxury = [
    'Aston Martin', 'Bentley', 'Ferrari', 'Lamborghini', 'Lexus', 'Lotus',
    'Maserati', 'McLaren', 'Porsche', 'Tesla']

premium = [
    'Audi', 'BMW', 'Cadillac', 'Infiniti', 'Land Rover',
    'MINI', 'Mercedes-Benz', 'Jaguar']

low_cost = ['Dacia', 'Skoda']

# Encode the segments to integer data
df['Segment'] = 2

mask = df.index.isin(luxury)
df.loc[mask, 'Segment'] = 4

mask = df.index.isin(premium)
df.loc[mask, 'Segment'] = 3

mask = df.index.isin(low_cost)
df.loc[mask, 'Segment'] = 1

# Assign each brand with each integer
df['Brand'] = df.index
df['Brand'] = df['Brand'].astype('category').cat.codes
df.head()

# %%
# 2.13.2. One-hot Encoding
df['Brand'] = df.index
df = pd.get_dummies(df, columns=['Brand'])
df.head()

# 2.13.3. Dataset Creation
# Define a function to split the dataset into train and test sets with a categorical column
def datasets_cat(df, x_len=12, y_len=1, test_loops=12, cat_name='_'):
    """
    Splits the dataframe into training and testing sets based on the specified
    lengths and test loops, considering categorical columns.
    """

    # Identify categorical columns and get dataset shape
    col_cat = [col for col in df.columns if cat_name in col]
    data_values = df.drop(columns=col_cat).values  # Historical demand
    categorical_values = df[col_cat].values  # Categorical info
    rows, periods = data_values.shape

    # Total number of loops (train + test)
    loops = periods + 1 - x_len - y_len

    # Create the training set
    train = [data_values[:, col:col + x_len + y_len] for col in range(loops)]
    train = np.vstack(train)
    X_train, Y_train = np.split(train, [-y_len], axis=1)
    X_train = np.hstack((np.vstack([categorical_values] * loops), X_train))

    # Split into train and test sets
    if test_loops > 0:
        X_train, X_test = np.split(X_train, [-rows * test_loops], axis=0)
        Y_train, Y_test = np.split(Y_train, [-rows * test_loops], axis=0)
    else:
        X_test = np.hstack((categorical_values, data_values[:, -x_len:]))
        Y_test = np.full((X_test.shape[0], y_len), np.nan)

    # Reshape Y_train and Y_test for scikit-learn compatibility
    if y_len == 1:
        Y_train = Y_train.ravel()
        Y_test = Y_test.ravel()

    return X_train, Y_train, X_test, Y_test

# %%
# Apply Integer Encoding
df = import_data()
df['Segment'] = 2

mask = df.index.isin(luxury)
df.loc[mask, 'Segment'] = 4

mask = df.index.isin(premium)
df.loc[mask, 'Segment'] = 3

mask = df.index.isin(low_cost)
df.loc[mask, 'Segment'] = 1

X_train, Y_train, X_test, Y_test = datasets_cat(
    df, x_len=12, y_len=1, test_loops=12, cat_name='Segment')

# %%
# Apply One-Hot Encoding
df['Brand'] = df.index
df = pd.get_dummies(df, columns=['Brand'], prefix_sep='_')

X_train, Y_train, X_test, Y_test = datasets_cat(
    df, x_len=12, y_len=1, test_loops=12, cat_name='_')

# %%
# 2.14. Clustering
# Define function to get the multiplicative seasonal factor for each period
def seasonal_factors(df, slen):
    s = pd.DataFrame(index=df.index)
    for i in range(slen):
        s[i + 1] = df.iloc[:, i::slen].mean(axis=1)

    s = s.divide(s.mean(axis=1), axis=0).fillna(0)
    return s

# Define function to Scale the seasonal factor to a range of 0 to 1
def scaler(s):
    mean = s.mean(axis=1)
    maxi = s.max(axis=1)
    mini = s.min(axis=1)
    s = s.subtract(mean, axis=0)
    s = s.divide(maxi - mini, axis=0).fillna(0)
    return s

# Apply for the dataset
df = import_data()
s = seasonal_factors(df,slen=12)
s = scaler(s)
print(s.head())

# %%
# from sklearn.cluster import KMeans

# Perform KMeans clustering with 4 clusters
kmeans = KMeans(n_clusters=4, random_state=0).fit(s)
df['Group'] = kmeans.predict(s)

# Evaluate KMeans with different cluster numbers
results = []
for n in range(1, 10):
    kmeans = KMeans(n_clusters=n, random_state=0).fit(s)
    results.append([n, kmeans.inertia_])

# Convert results to DataFrame and plot
results = pd.DataFrame(
    data=results, columns=['Number of clusters', 'Inertia']
).set_index('Number of clusters')

results.plot()

# %%
import calendar
import seaborn as sns

# Perform KMeans clustering with 4 clusters
kmeans = KMeans(n_clusters=4, random_state=0).fit(s)

# Create a DataFrame for cluster centers
centers = pd.DataFrame(data=kmeans.cluster_centers_).transpose()
centers.index = calendar.month_abbr[1:]
centers.columns = [f'Cluster {x}' for x in range(centers.shape[1])]

# Plot heatmap of cluster centers
sns.heatmap(centers, annot=True, fmt='.2f', center=0, cmap='RdBu_r')

# Print value counts of each group
print(df['Group'].value_counts().sort_index())

# %%
# 2.15. Feature Optimization #2 
def datasets_full(
    df, X_exo, x_len=12, y_len=1, test_loops=12, holdout_loops=0, cat_name=['_']):
    '''
    Generates training, holdout, and test datasets for time series forecasting.

    Parameters:
    df (pd.DataFrame): DataFrame containing historical demand data.
    X_exo (np.array): Exogenous variables affecting demand.
    x_len (int): Number of past periods used as features (default: 12).
    y_len (int): Forecast horizon (default: 1).
    test_loops (int): Number of test samples (default: 12).
    holdout_loops (int): Number of holdout samples (default: 0).
    cat_name (list): List of substrings indicating categorical columns (default: ['_']).

    Returns:
    tuple: (X_train, Y_train, X_holdout, Y_holdout, X_test, Y_test, features)
    '''

    # Identify categorical columns based on specified substrings in column names
    col_cat = [col for col in df.columns if any(name in col for name in cat_name)]
    categorical_values = df[col_cat].values  # Extract categorical data
    data_values = df.drop(columns=col_cat).values  # Extract numerical demand data
    rows, periods = data_values.shape  # Number of rows (items) and periods (time steps)

    # Repeat exogenous variables for each row in the dataset
    X_exo = np.repeat(np.reshape(X_exo, [1, -1]), rows, axis=0)

    # Extract month information from column names (assumed last 2 characters represent the month)
    X_months = np.repeat(
        np.reshape(
            [int(col[-2:]) for col in df.columns if col not in col_cat], [1, -1]
        ),
        rows,
        axis=0,)

    # Training set creation
    loops = periods + 1 - x_len - y_len  # Number of rolling windows
    train = []

    for col in range(loops):
        m = X_months[:, col + x_len].reshape(-1, 1)  # Extract month as a feature
        exo = X_exo[:, col : col + x_len + y_len]  # Select exogenous variables

        # Aggregate exogenous features
        exo = np.hstack(
            [
                np.mean(exo, axis=1, keepdims=True),  # Mean of all exogenous data
                np.mean(exo[:, -4:], axis=1, keepdims=True),  # Mean of last 4 months
                exo,
            ])

        d = data_values[:, col : col + x_len + y_len]  # Extract demand data

        # Aggregate demand features
        d = np.hstack(
            [
                np.mean(d[:, :-y_len], axis=1, keepdims=True),  # Mean demand
                np.median(d[:, :-y_len], axis=1, keepdims=True),  # Median demand
                np.mean(d[:, -4 - y_len : -y_len], axis=1, keepdims=True),  # 4-month MA
                np.max(d[:, :-y_len], axis=1, keepdims=True),  # Max demand
                np.min(d[:, :-y_len], axis=1, keepdims=True),  # Min demand
                d,])

        # Append all features to the training dataset
        train.append(np.hstack([m, exo, d]))

    train = np.vstack(train)  # Stack training samples into a single array
    X_train, Y_train = np.split(train, [-y_len], axis=1)  # Split features and target

    # Include categorical values in the feature matrix
    X_train = np.hstack((np.vstack([categorical_values] * loops), X_train))

    # Define feature names
    features = (
        col_cat
        + ['Month']
        + ['Exo Mean', 'Exo MA4']
        + [f'Exo M{-x_len+col}' for col in range(x_len + y_len)]
        + [
            'Demand Mean',
            'Demand Median',
            'Demand MA4',
            'Demand Max',
            'Demand Min',]
        + [f'Demand M-{x_len-col}' for col in range(x_len)])

    # Holdout set creation
    if holdout_loops > 0:
        X_train, X_holdout = np.split(X_train, [-rows * holdout_loops], axis=0)
        Y_train, Y_holdout = np.split(Y_train, [-rows * holdout_loops], axis=0)
    else:
        X_holdout, Y_holdout = np.array([]), np.array([])

    # Test set creation
    if test_loops > 0:
        X_train, X_test = np.split(X_train, [-rows * test_loops], axis=0)
        Y_train, Y_test = np.split(Y_train, [-rows * test_loops], axis=0)
    else:  # No test set: X_test is used to generate future forecasts
        exo = X_exo[:, -x_len - y_len :]
        d = data_values[:, -x_len:]

        X_test = np.hstack(
            (
                categorical_values,
                m[:, -1].reshape(-1, 1),  # Latest available month
                np.hstack(
                    [
                        np.mean(exo, axis=1, keepdims=True),
                        np.mean(exo[:, -4:], axis=1, keepdims=True),
                        exo,]),
                np.hstack(
                    [
                        np.mean(d, axis=1, keepdims=True),
                        np.median(d, axis=1, keepdims=True),
                        np.mean(d[:, -4:], axis=1, keepdims=True),
                        np.max(d, axis=1, keepdims=True),
                        np.min(d, axis=1, keepdims=True),
                        d,]),))
        Y_test = np.full((X_test.shape[0], y_len), np.nan)  # Dummy values for prediction

    # Format target variables for scikit-learn (flatten if y_len = 1)
    if y_len == 1:
        Y_train = Y_train.ravel()
        Y_test = Y_test.ravel()
        Y_holdout = Y_holdout.ravel()

    return X_train, Y_train, X_holdout, Y_holdout, X_test, Y_test, features

# Import dataset
df = import_data()

# Load GDP data and set 'Year' as the index
GDP = pd.read_excel('GDP.xlsx').set_index('Year')

# Extract year information from the column names of df
dates = pd.to_datetime(df.columns, format='%Y-%m').year

# Map GDP values to corresponding years in df
X_GDP = [GDP.loc[date, 'GDP'] for date in dates]

# Define vehicle brand segments
luxury = [
    'Aston Martin', 'Bentley', 'Ferrari', 'Lamborghini', 'Lexus', 'Lotus',
    'Maserati', 'McLaren', 'Porsche', 'Tesla']

premium = [
    'Audi', 'BMW', 'Cadillac', 'Infiniti', 'Land Rover',
    'MINI', 'Mercedes-Benz', 'Jaguar']

low_cost = ['Dacia', 'Skoda']

# Default all brands to segment 2
df['Segment'] = 2

# Assign segment values based on brand category
df.loc[df.index.isin(luxury), 'Segment'] = 4
df.loc[df.index.isin(premium), 'Segment'] = 3
df.loc[df.index.isin(low_cost), 'Segment'] = 1

# Store brand names in a new column
df['Brand'] = df.index

# Convert 'Brand' into one-hot encoded features
df = pd.get_dummies(df, columns=['Brand'], prefix_sep='_')


from sklearn.model_selection import train_test_split

# Generate datasets for training, holdout, and testing
X_train, Y_train, X_holdout, Y_holdout, X_test, Y_test, features = datasets_full(
    df, X_GDP, x_len=12, y_len=1, test_loops=12, holdout_loops=0,
    cat_name=['_', 'Segment', 'Group']
)

# Split the training dataset into training and validation sets (15% for validation)
x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.15)


from xgboost.sklearn import XGBRegressor

# Initialize XGBoost Regressor with specified hyperparameters
XGB = XGBRegressor(
    n_jobs=-1,
    max_depth=10,
    n_estimators=1000,
    learning_rate=0.01,
    objective='reg:absoluteerror',
    early_stopping_rounds=100
)

# Train the model using the validation set for early stopping evaluation
XGB = XGB.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)

# Make predictions on training and test sets
Y_train_pred = XGB.predict(X_train)
Y_test_pred = XGB.predict(X_test)

# Evaluate model performance
kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name='XGBoost')


# Get feature importance from the trained XGBoost model
imp = XGB.get_booster().get_score(importance_type='total_gain')

# Convert importance dictionary to a DataFrame
imp = pd.DataFrame.from_dict(imp, orient='index', columns=['Importance'])

# Map feature indices to actual feature names
imp.index = np.array(features)[
    imp.index.astype(str).str.replace('f', '').astype(int)
]

# Normalize importance values and sort in descending order
imp = (imp['Importance'] / sum(imp.values)).sort_values(ascending=False)

# Save feature importance to an Excel file
imp.to_excel('Feature Importance.xlsx')

# Display the top features
imp.head()


def model_kpi(model, X, Y):
    """
    Calculate MAE and RMSE as a percentage of the mean actual values.

    Parameters:
    model: Trained model with a predict method.
    X (array-like): Feature matrix.
    Y (array-like): True target values.

    Returns:
    tuple: (MAE, RMSE) as relative error percentages.
    """
    Y_pred = model.predict(X)
    mae = np.mean(np.abs(Y - Y_pred)) / np.mean(Y)
    rmse = np.sqrt(np.mean((Y - Y_pred) ** 2)) / np.mean(Y)

    return mae, rmse


# Initialize an empty list to store results
results = []

# Define the list of limits for filtering features
limits = [
    0.00005, 0.0001, 0.00015, 0.0002, 0.00025, 0.0003, 0.0004,
    0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.0011, 0.002, 0.004,
    0.008, 0.01, 0.02, 0.04, 0.06
]

# Initialize the XGBRegressor model with specific hyperparameters
XGB = XGBRegressor(
    n_jobs=-1,                    # Use all available cores for parallelism
    max_depth=10,                 # Maximum depth of the trees
    n_estimators=1000,            # Number of boosting rounds
    learning_rate=0.01,           # Step size shrinking
    objective='reg:absoluteerror', # Objective function
    early_stopping_rounds=100     # Stop early if no improvement after 100 rounds
)

# Iterate over each limit to filter features and train the model
for limit in limits:
    # Create a mask to filter features based on importance
    mask = [feature in imp[imp > limit] for feature in features]

    # Train the model using the filtered features
    XGB = XGB.fit(
        x_train[:, mask],
        y_train,
        verbose=False,
        eval_set=[(x_val[:, mask], y_val)]
    )

    # Append the model performance metrics to the results
    results.append(model_kpi(XGB, x_val[:, mask], y_val))

# Convert results into a DataFrame with proper columns and index
results = pd.DataFrame(data=results, columns=['MAE', 'RMSE'], index=limits)

# Plot the results with MAE on the secondary y-axis and log scale on x-axis
results.plot(secondary_y='MAE', logx=True)


# Define the limit for filtering the importance values
limit = 0.007

# Print the index of features with importance greater than the limit
print(imp[imp > limit].index)


# Create a mask to filter features based on importance
mask = [feature in imp[imp > limit] for feature in features]

# Train the XGB model using the filtered features
XGB = XGB.fit(
    x_train[:, mask],
    y_train,
    verbose=False,
    eval_set=[(x_val[:, mask], y_val)]
)

# Predict the target variable for training and test datasets
Y_train_pred = XGB.predict(X_train[:, mask])
Y_test_pred = XGB.predict(X_test[:, mask])

# Calculate and print the performance metrics using KPI function
kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name='XGBoost')


# %%
# 2.16. Neural Network

from sklearn.neural_network import MLPRegressor
NN = MLPRegressor().fit(X_train, Y_train)

# Neural Network Parameters
hidden_layer_sizes = [
    [neuron] * hidden_layer
    for neuron in range(10, 60, 10)
    for hidden_layer in range(2, 7)]
alpha = [5, 1, 0.5, 0.1, 0.05, 0.01, 0.001]
learning_rate_init = [0.05, 0.01, 0.005, 0.001, 0.0005]
beta_1 = [0.85, 0.875, 0.9, 0.95, 0.975, 0.99, 0.995]
beta_2 = [0.99, 0.995, 0.999, 0.9995, 0.9999]
param_dist = {
    'hidden_layer_sizes': hidden_layer_sizes,
    'alpha': alpha,
    'learning_rate_init': learning_rate_init,
    'beta_1': beta_1,
    'beta_2': beta_2}

# Adam Parameters
activation = 'relu'
solver = 'adam'
early_stopping = True
n_iter_no_change = 50
validation_fraction = 0.1
tol = 0.0001

param_fixed = {
    'activation': activation,
    'solver': solver,
    'early_stopping': early_stopping,
    'n_iter_no_change': n_iter_no_change,
    'validation_fraction': validation_fraction,
    'tol': tol}

# Run NN with Adam optimizer
NN = MLPRegressor(hidden_layer_sizes=(20,20), **param_fixed, verbose=True).fit(X_train, Y_train)

# Using Random Search to test NN parameter to find the best model
NN = MLPRegressor(**param_fixed)
NN_cv = RandomizedSearchCV(NN, param_dist, cv=10, verbose=2, n_jobs=-1, n_iter=200, scoring='neg_mean_absolute_error')
NN_cv.fit(X_train, Y_train)
print('Tuned NN Parameters:', NN_cv.best_params_)
print()
Y_train_pred = NN_cv.predict(X_train)
Y_test_pred = NN_cv.predict(X_test)
kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name='NN optimized')