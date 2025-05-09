lst = [10, 507, 'python']
print("created list is:", lst)

lst.append(20)
lst.append("program")
lst.append([3, 7])
print("After adding elements the new list is:", lst)

lst.pop()      
lst.pop(2)     
lst.remove(10) 
print("After deleting elements the new list is:", lst)

lst.clear()
print("After removing all the elements the new list is:", lst)
