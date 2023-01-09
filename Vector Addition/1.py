import numpy as np
import matplotlib.pyplot as plt
data = [[8, 17, 32, 61, 123, 246, 493],
		[0, 0, 0, 0, 0, 0, 0],
		[4, 9, 18, 38, 83, 155, 306]]
X = np.arange(len(data[0]))
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(X + 0.00, data[0], color = 'r', width = 0.25)
ax.bar(X + 0.25, data[1], color = 'g', width = 0.25)
ax.bar(X + 0.50, data[2], color = 'b', width = 0.25)
ax.legend(labels=['Host to Device', 'Kernel', 'Device to Host'])
ax.set_xticks(X)
ax.set_xticklabels([1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864])
plt.xlabel('Number of input values')
plt.ylabel('Execution time (ms)')
fig.tight_layout()
plt.show()
