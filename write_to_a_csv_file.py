import csv

with open("dataW.csv", "w", newline="") as file: # newline="" prevents adding extra blank lines between rows on some systems.
  writer = csv.writer(file)
  writer.writerow(["Name", "Age", "City"]) # write header
  writer.writerow(["Alice", 30, "New York"])
  writer.writerow(["Bob", 25, "Los Angeles"])
