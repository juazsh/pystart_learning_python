import csv
with open("sample_data.csv", "r") as file:
  reader = csv.reader(file)
  for row in reader:
    print(row)
  print(reader)
  for row in reader:
    print("new" + row[2])