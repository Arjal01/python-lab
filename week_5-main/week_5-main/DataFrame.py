import pandas as pd

data = {'Name': ['John', 'Alice', 'Bob', 'Eve', 'Charlie'],
        'Age': [25, 30, 22, 35, 28],
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'Salary': [50000, 55000, 40000, 70000, 48000]}

df = pd.DataFrame(data)

# Displaying the Entire DataFrame
print("1. Entire DataFrame:")
print(df)
print("\n")

# Accessing Columns From DataFrame
print("2. Age Column:")
print(df['Age'])
print("\n")

# Accessing Rows by Index
print("3. Second Row:")
print(df.iloc[1])
print("\n")

# Accessing Multiple Rows or Columns
print("4. First Two Rows (Name, Age):")
print(df.loc[0:1, ['Name', 'Age']])
print("\n")

# Accessing Rows Based on Conditions
print("5. Age > 25:")
print(df[df['Age'] > 25])
print("\n")

# Accessing Specific Cells with at and iat
print("6. Salary at Index 1:")
print(df.at[1, 'Salary'])