import matplotlib.pyplot as plt

# Your Pi estimations
nums = [1_822_725, 22_059_421, 32_374_695, 88_754_320, 97_162_66, 200_745_654]

# Numba-optimized results
numba_results = [3.1415935153733052, 3.1415927247955033, 3.1415927021077117, 3.141592635888531, 3.1415924919219385, 3.141592645761164]

# Correct value of Pi
correct_pi = 3.141592653589793

# Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(nums, numba_results, label='Numba-optimized', marker='o', color='blue')
plt.axhline(y=correct_pi, color='r', linestyle='--', label='Correct Pi')

# Set logarithmic scale for x-axis
plt.xscale('log')

plt.xlabel('Number of Iterations (N)')
plt.ylabel('Approximated Pi')
plt.title('Scatter Plot of Pi Approximations with Numba Optimization')
plt.legend()
plt.show()
