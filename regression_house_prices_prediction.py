from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

#load dataset 
california = fetch_california_housing()
df = pd.DataFrame(california.data, columns=california.feature_names)
df['PRICE'] = california.target # Add the target column for house prices

print(df.head())

# summary statistics
print(df.describe())

## check for missing values
print(df.isnull().sum())

# preprocessing
## handle missing values (if any): use mthods like filling with eman values or removingr rows.
## split data: divide the data into training and test sets
## standardize featuers: many regression models perform better if data is scaled

X = df.drop("PRICE", axis=1)
y = df["PRICE"]

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)


# evaluate the model
## to measure the model's accuracy, we can use metrics like Mean Squared Error (MSE) and R-squared.

#make predictions
y_pred = model.predict(X_test)

#calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error!", mse)
print("R-squared:", r2)
