import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv("house_price.csv")

print("5 dòng đầu:")
print(data.head())
print("\nKích thước dữ liệu:")
print(data.shape)


X = data[
    [
        "Area_m2",
        "Bedrooms",
        "Distance_Center_km"
    ]
]
y = data["Price_Billion_VND"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print("\nSố dữ liệu train:", len(X_train))
print("Số dữ liệu test:", len(X_test))

# TẠO OVERFITTING
# Polynomial bậc 8
overfit_model = make_pipeline(
    PolynomialFeatures(degree=8),
    LinearRegression()
)
overfit_model.fit(X_train, y_train)

# Dự đoán
y_train_pred = overfit_model.predict(X_train)
y_test_pred = overfit_model.predict(X_test)

# KIỂM TRA OVERFITTING
train_mse_before = mean_squared_error(
    y_train,
    y_train_pred
)
test_mse_before = mean_squared_error(
    y_test,
    y_test_pred
)
print("\n========== TRƯỚC KHI DÙNG K-FOLD ==========")
print("Train MSE:", train_mse_before)
print("Test MSE :", test_mse_before)
if train_mse_before < test_mse_before:
    print("=> Có dấu hiệu Overfitting")
else:
    print("=> Chưa thấy Overfitting rõ ràng")

#  K-FOLD CROSS VALIDATION
kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
degrees = [1, 2, 3, 4, 5, 6, 7, 8]
results = []
print("\n========== K-FOLD ==========")
for degree in degrees:
    model = make_pipeline(
        PolynomialFeatures(degree=degree),
        LinearRegression()
    )
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=kf,
        scoring="neg_mean_squared_error"
    )
    mean_mse = -scores.mean()
    results.append({
        "Degree": degree,
        "MSE": mean_mse
    })
    print(
        "Bậc",
        degree,
        "-> MSE:",
        mean_mse
    )

#CHỌN BẬC TỐT NHẤT
results_df = pd.DataFrame(results)
best_degree = results_df.loc[
    results_df["MSE"].idxmin(),
    "Degree"
]
print("\nBậc Polynomial tốt nhất:", best_degree)

#TẠO MÔ HÌNH SAU KHI KHẮC PHỤC
fixed_model = make_pipeline(
    PolynomialFeatures(
        degree=int(best_degree)
    ),
    LinearRegression()
)
fixed_model.fit(X_train, y_train)

#DỰ ĐOÁN SAU KHI KHẮC PHỤC
y_train_pred_fixed = fixed_model.predict(X_train)
y_test_pred_fixed = fixed_model.predict(X_test)

#TÍNH MSE SAU KHI KHẮC PHỤC
train_mse_after = mean_squared_error(
    y_train,
    y_train_pred_fixed
)
test_mse_after = mean_squared_error(
    y_test,
    y_test_pred_fixed
)
print("\n========== SAU KHI DÙNG K-FOLD ==========")
print("Train MSE:", train_mse_after)
print("Test MSE :", test_mse_after)

# SO SÁNH
print("\n========== SO SÁNH ==========")
print("Trước K-Fold:")
print("Train MSE:", train_mse_before)
print("Test MSE :", test_mse_before)

print("\nSau K-Fold:")
print("Train MSE:", train_mse_after)
print("Test MSE :", test_mse_after)

#DỰ ĐOÁN MỘT CĂN NHÀ MỚI
new_house = pd.DataFrame({
    "Area_m2": [100],
    "Bedrooms": [3],
    "Distance_Center_km": [5]
})
predicted_price = fixed_model.predict(new_house)
print("\n========== DỰ ĐOÁN NHÀ MỚI ==========")
print(
    "Giá nhà dự đoán:",
    round(predicted_price[0], 2),
    "tỷ VNĐ"
)