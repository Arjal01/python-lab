import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [50, 65, 85, 87, 98]
text = ["IBM", "Amazon", "Facebook", "Microsoft", "Google"]
colors = ["red", "orange", "yellow", "blue", "green"]

plt.xlim(0, 6)
plt.ylim(0, 100)
plt.bar(x, y, tick_label=text, color=colors, linewidth=0.5)
plt.xlabel("Company")
plt.ylabel("Percentage")
plt.title("Percentage Graph")
plt.show()