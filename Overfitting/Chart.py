import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

#ĐỌC DỮ LIỆU
data = pd.read_csv("house_price.csv")

X = data[["Area_m2", "Bedrooms", "Distance_Center_km"]]
y = data["Price_Billion_VND"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#TÌM BẬC TỐT NHẤT QUA K-FOLD
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_degree = 1
best_mse = float("inf")

for degree in range(1, 9):
    model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
    scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="neg_mean_squared_error")
    mean_mse = -scores.mean()
    if mean_mse < best_mse:
        best_mse = mean_mse
        best_degree = degree

# HUẤN LUYỆN 2 MÔ HÌNH (OVERFIT VÀ K-FOLD TỐI ƯU)
# Mô hình Overfit bậc 8
model_overfit = make_pipeline(PolynomialFeatures(degree=8), LinearRegression())
model_overfit.fit(X_train, y_train)

# Mô hình sau khi tối ưu bậc qua K-Fold
model_kfold = make_pipeline(PolynomialFeatures(degree=int(best_degree)), LinearRegression())
model_kfold.fit(X_train, y_train)

#TẠO DẢI ĐIỂM ĐỂ VẼ ĐƯỜNG CONG DỰ ĐOÁN
area_range = np.linspace(X["Area_m2"].min(), X["Area_m2"].max(), 300)
grid_data = pd.DataFrame({
    "Area_m2": area_range,
    "Bedrooms": X_train["Bedrooms"].median(),          # Cố định ở trung vị
    "Distance_Center_km": X_train["Distance_Center_km"].mean() # Cố định ở trung bình
})

y_curve_overfit = model_overfit.predict(grid_data)
y_curve_kfold = model_kfold.predict(grid_data)

# VẼ BIỂU ĐỒ CHUẨN MẪU
plt.figure(figsize=(10, 6))

# Điểm dữ liệu huấn luyện
plt.scatter(X_train["Area_m2"], y_train, color="black", label="Training data", zorder=5)

# Đường cong Overfit (Đỏ - bậc 8 dao động mạnh)
plt.plot(area_range, y_curve_overfit, color="red", label="Overfit (Degree: 8)", linewidth=2)

# Đường cong K-Fold (Xanh lá - khái quát hóa tốt)
plt.plot(area_range, y_curve_kfold, color="green", label=f"With K-fold tuning (Degree: {best_degree})", linewidth=3)

# Giới hạn trục Y để tránh đường bậc 8 bùng nổ làm mất góc nhìn
y_min = y_train.min() - 1
y_max = y_train.max() + 2
plt.ylim(y_min, y_max)

plt.title("Overfitting and K-fold CV tuning", fontsize=13, fontweight="bold")
plt.xlabel("Area_m2 (Diện tích)")
plt.ylabel("Price_Billion_VND (Giá nhà)")
plt.legend(loc="upper left")
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()