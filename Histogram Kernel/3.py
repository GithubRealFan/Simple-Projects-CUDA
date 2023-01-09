from matplotlib import pyplot as plt
import numpy as np

with open('3.csv', 'r') as fp:
	data = [int(x.strip()) for x in fp.readlines()]
data = np.array(data)

X = np.arange(len(data))

fig = plt.figure(figsize = (8, 4))

# creating the bar plot
plt.bar(X, data, width=0.5)

plt.xlabel('Value')
plt.ylabel('Count')
 
# Show plot
plt.show()
