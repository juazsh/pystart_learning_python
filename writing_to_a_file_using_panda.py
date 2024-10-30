import pandas as pd

#create a DataFrame
data = {
  "Name": ["Alice", "Bob"],
  "Age": [30, 25],
  "City": ["New York", "Los Angeles"]
}
df = pd.DataFrame(data)

# write dataFrame to csv
df.to_csv("data.csv", index=False) # index=False avoids adding a row index