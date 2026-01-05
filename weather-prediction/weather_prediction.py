import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Create sample weather dataset
data = {
    "temperature": [25, 28, 30, 35, 40, 42, 45],
    "humidity": [70, 65, 60, 55, 50, 45, 40],
    "rainfall": [5, 4, 3, 2, 1, 0, 0]
}

df = pd.DataFrame(data)

X = df[["temperature", "humidity"]]
y = df["rainfall"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

prediction = model.predict([[32, 58]])
print("Predicted rainfall (mm):", round(prediction[0], 2))
