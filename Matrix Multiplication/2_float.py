import numpy as np
import matplotlib.pyplot as plt
data = [[0, 0, 1, 1, 3, 13, 58],
		[0, 0, 0, 0, 0, 0, 0],
		[0, 1, 1, 10, 71, 554, 2816]]
X = np.arange(7)
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(X + 0.00, data[0], color = 'r', width = 0.25)
ax.bar(X + 0.25, data[1], color = 'g', width = 0.25)
ax.bar(X + 0.50, data[2], color = 'b', width = 0.25)
ax.legend(labels=['Host to Device', 'Kernel', 'Device to Host'])
ax.set_xticks(X)
ax.set_xticklabels([64, 128, 256, 512, 1024, 2048, 4096])
plt.xlabel('Matrix size')
plt.ylabel('Execution time (ms)')
fig.tight_layout()
plt.show()
