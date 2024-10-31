import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = {
  "YearsExperience": [1,2,3,4,5],
  "Age": [22, 24, 26, 28, 30]
}
df = pd.DataFrame(data)

X = df[["YearsExperience"]]
y = df[["Age"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

# create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#evaluate the model
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error: ", mse)

