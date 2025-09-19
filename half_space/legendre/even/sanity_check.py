from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from npeet import entropy_estimators as ee
import numpy as np

# Generate synthetic data
X, Y = make_regression(n_samples=1000, n_features=4, noise=0.1, random_state=42)
Y = Y.reshape(-1, 1)  # Make it 2D

# Encode: X -> Y is known
# Decode: train regression Y -> X
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train decoder
reg = LinearRegression()
reg.fit(Y_train, X_train)
Xhat = reg.predict(Y_test)

# Normalize
scaler_X = StandardScaler().fit(X_train)
scaler_Y = StandardScaler().fit(Y_train)

X_test_n = scaler_X.transform(X_test)
Y_test_n = scaler_Y.transform(Y_test)
Xhat_n = scaler_X.transform(Xhat)

# Estimate MIs
I_XY = ee.mi(X_test_n, Y_test_n)
I_XXhat = ee.mi(X_test_n, Xhat_n)

print("I(X; Y)    =", I_XY)
print("I(X; Xhat) =", I_XXhat)
