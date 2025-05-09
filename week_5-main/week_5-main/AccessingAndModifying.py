import pandas as pd

data = {'Name': ['John', 'Alice', 'Bob', 'Eve', 'Charlie'],
        'Age': [25, 30, 22, 35, 28],
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'Salary': [50000, 55000, 40000, 70000, 48000]}

df = pd.DataFrame(data)

# Accessing the Index
print("1. Accessing the Index:")
print(df.index)
print("\n")

# Setting a Custom Index
df_with_index = df.set_index('Name')
print("2. Setting a Custom Index ('Name'):")
print(df_with_index)
print("\n")

# Resetting the Index
df_reset = df_with_index.reset_index()
print("3. Resetting the Index:")
print(df_reset)
print("\n")

# Indexing with loc
row = df_with_index.loc['Alice']
print("4. Indexing with loc (Row for 'Alice'):")
print(row)
print("\n")

# Changing the Index
df_with_new_index = df.set_index('Age')
print("5. Changing the Index ('Age'):")
print(df_with_new_index)