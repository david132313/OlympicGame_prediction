import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
data = {
    "Year": [1904, 1932, 1984, 1996],
    "Host": ["United States"] * 4,
    "Total_Predicted": [53.529670, 68.808127, 97.182406, 103.730316],
    "Total": [231.0, 110.0, 174.0, 101.0],
    "Difference_Total": [177.470330, 41.191873, 76.817594, -2.730316],
    "Weighted_Total_Predicted": [37.347206, 46.145694, 62.485742, 66.256523],
    "Total_weighted": [140.666667, 72.0, 123.5, 68.333333],
    "Difference_Weighted_Total": [103.319460, 25.854306, 61.014258, 2.076811],
}
# Convert data to DataFrame
df = pd.DataFrame(data)

# Create X values (year-1903)
X = df['Year'] - 1903
X_log = np.log(X)

# Fit logarithmic regression for Difference_Total
coeffs_total = np.polyfit(X_log, df['Difference_Total'], 1)
log_fit_total = lambda x: coeffs_total[0] * np.log(x) + coeffs_total[1]

# Fit logarithmic regression for Difference_Weighted_Total
coeffs_weighted = np.polyfit(X_log, df['Difference_Weighted_Total'], 1)
log_fit_weighted = lambda x: coeffs_weighted[0] * np.log(x) + coeffs_weighted[1]

# Calculate R-squared values
y_pred_total = log_fit_total(X)
r2_total = r2_score(df['Difference_Total'], y_pred_total)

y_pred_weighted = log_fit_weighted(X)
r2_weighted = r2_score(df['Difference_Weighted_Total'], y_pred_weighted)

# Create functions to make predictions
def predict_difference_total(year):
    x = year - 1903
    return log_fit_total(x)

def predict_difference_weighted(year):
    x = year - 1903
    return log_fit_weighted(x)

# Print model equations and R-squared values
print("Model 1: Difference_Total")
print(f"Equation: y = {coeffs_total[0]:.2f} * ln(year-1903) + {coeffs_total[1]:.2f}")
print(f"R-squared: {r2_total:.4f}\n")

print("Model 2: Difference_Weighted_Total")
print(f"Equation: y = {coeffs_weighted[0]:.2f} * ln(year-1903) + {coeffs_weighted[1]:.2f}")
print(f"R-squared: {r2_weighted:.4f}\n")

# Create visualization
plt.figure(figsize=(12, 6))

# Plot Difference_Total
plt.subplot(1, 2, 1)
plt.scatter(X, df['Difference_Total'], color='blue', label='Actual')
X_smooth = np.linspace(X.min(), X.max(), 100)
plt.plot(X_smooth, log_fit_total(X_smooth), color='red', label='Predicted')
plt.xlabel('Years since 1903')
plt.ylabel('Difference_Total')
plt.title('Logarithmic Regression: Difference_Total')
plt.legend()

# Plot Difference_Weighted_Total
plt.subplot(1, 2, 2)
plt.scatter(X, df['Difference_Weighted_Total'], color='blue', label='Actual')
plt.plot(X_smooth, log_fit_weighted(X_smooth), color='red', label='Predicted')
plt.xlabel('Years since 1903')
plt.ylabel('Difference_Weighted_Total')
plt.title('Logarithmic Regression: Difference_Weighted_Total')
plt.legend()

plt.tight_layout()
#plt.show()

# Example predictions for a future year
future_year = 2028
print(f"Predictions for {future_year}:")
print(f"Predicted Difference_Total: {predict_difference_total(future_year):.2f}")
print(f"Predicted Difference_Weighted_Total: {predict_difference_weighted(future_year):.2f}")
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np


data2 = {
    "Year": [1956, 2000],
    "Difference_Total": [18.939629, 26.072952],
    "Difference_Weighted_Total": [12.221565, 15.508648]
}

# Converting into a DataFrame
australia_df = pd.DataFrame(data2)

# Adding a column for the years since 1950 (to normalize and make regression easier)
australia_df["Years_Since_1950"] = australia_df["Year"] - 1950

# Preparing the regression models for both columns
X = australia_df[["Years_Since_1950"]]
y_total_diff = australia_df["Difference_Total"]
y_weighted_diff = australia_df["Difference_Weighted_Total"]

# Training the models
model_total_diff = LinearRegression().fit(X, y_total_diff)
model_weighted_diff = LinearRegression().fit(X, y_weighted_diff)

# Predicting for 2032 (Years_Since_1950 = 2032 - 1950)
future_year = 2032 - 1950
predicted_total_diff = model_total_diff.predict([[future_year]])
predicted_weighted_diff = model_weighted_diff.predict([[future_year]])

print(predicted_total_diff[0], predicted_weighted_diff[0])