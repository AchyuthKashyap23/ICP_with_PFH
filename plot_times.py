import numpy as np 
import matplotlib.pyplot as plt

# Varying the number of neighbors
# neighbors: 3
# 13540.963752087484
# Time taken for ICP: 211.40703082084656
# neighbors: 5
# 7.637541376128477e-25
# Time taken for ICP: 204.55474090576172
# neighbors: 10
# 7.637541376128477e-25
# Time taken for ICP: 216.50944304466248
# Varying the number of bins
# bins: 1
# 9300269.447835976
# Time taken for ICP: 48.191879987716675
# bins: 3
# 65.94993005194834
# Time taken for ICP: 51.64779591560364
# bins: 5
# 7.637541376128477e-25
# Time taken for ICP: 60.5616409778595
# bins: 10
# 7.637541376128477e-25
# Time taken for ICP: 199.3337209224701


time_neighbors = [211.40703082084656, 204.55474090576172, 216.50944304466248]

time_bins = [48.191879987716675, 51.64779591560364, 60.5616409778595, 199.3337209224701]

fig, ax = plt.subplots()
ax.plot([3, 5, 10], time_neighbors)
ax.set_xlabel('Number of neighbors')
ax.set_ylabel('Time taken')
plt.title('Time taken vs Number of neighbors')
plt.savefig('plots/cat_rotated/time_neighbors.png')

fig, ax = plt.subplots()
ax.plot([1, 3, 5, 10], time_bins)
ax.set_xlabel('Number of bins')
ax.set_ylabel('Time taken')
plt.title('Time taken vs Number of bins')
plt.savefig('plots/cat_rotated/time_bins.png')
