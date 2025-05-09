import pandas as pd

names = ['Alice', 'Bob', 'Charlie']
ages = [25, 30, 22]
professions = ['Engineer', 'Doctor', 'Artist']

data = {
    'Name': names,
    'Age': ages,
    'Profession': professions
}

df = pd.DataFrame(data)

print(df)
