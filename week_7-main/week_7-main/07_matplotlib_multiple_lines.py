import matplotlib.pyplot as plt

x1 = [1, 4, 6, 8]
y1 = [2, 5, 8, 9]
x2 = [3, 6, 8, 10]
y2 = [2, 4, 8, 9]

plt.plot(x1, y1, label="line A", color="r")
plt.plot(x2, y2, label="line B", color="g")
plt.xlim(0, 12)
plt.ylim(0, 12)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Graph")
plt.legend()
plt.show()