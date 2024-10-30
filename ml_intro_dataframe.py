import pandas as pd
import matplotlib.pyplot as plt

#load scv into a DataFrame
df = pd.read_csv("sample_data.csv")
print(df)
print(df.head()) # By default, shows the first 5 rows
print(df.info()) #Displays column names, data types, and non-null counts
print(df.describe()) # shows stats like mean, min, max for numeric columns

# filtering
over_30 = df[df["Age"] > 30]
print(over_30)

# check for missing values
print(df.isnull().sum())
# fill missing values:
df.fillna("Unknown", inplace=True) # Replace nulls with "Unknown"

# aggregating data : summing averaging and counting values
#average 
avg_age = df["Age"].mean()
print("Average Age:", avg_age)
#total count per city
city_counts = df["City"].value_counts()
print(city_counts)

#Grouping Data
# group by 'City' and calculate the mean age
avg_age_per_city = df.groupby("City")["Age"].mean()
print(avg_age_per_city)

# basic visualizatin with matlib
city_counts.plot(kind="bar")
plt.title("Number of People per City")
plt.xlabel("City")
plt.ylabel("Count")
plt.show()
