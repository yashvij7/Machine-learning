import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.array([1,2,2.5,3,3.5,4,4.5,5,5.5,6,
              6.5,7,7.5,8,8.5,9,9.5,10,10.5,11]).reshape(-1,1)

y = np.array([32,35,40,45,48,50,55,60,63,67,
              70,72,78,82,85,88,92,95,98,100])

print("Independent Variable (Study Hours):")
print(X)

print("\nDependent Variable (Score):")
print(y)

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

print("\nSlope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)

plt.scatter(X, y)
plt.plot(X, y_pred)
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Simple Linear Regression - Bigger Dataset")
plt.show()