print("Empty dictionary is:", {})

d = {1: 'apple', 2: 'ball'}
print("dictionary with integer keys", d)

d = {'name': 'rishi', 1: [2, 4, 3]}
print("dictionary with mixed keys", d)

d = dict.fromkeys("abcd", 'alphabet')
print("dictionary created by using dict.fromkeys method=", d)

d = {'name': 'jack', 'age': 25}
print(d['name'])  # Output: jack

d['age'] = 18
d['class'] = "B.Tech"
print("After changing and adding the values,the new dictionary=", d)

print("items in the dictionary is:", d.items())
print("Keys in the dictionary is:", d.keys())
print("values in the dictionary is:", d.values())
