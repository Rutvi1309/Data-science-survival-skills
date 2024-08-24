import numpy as np
import time
from numba import jit

@jit(nopython=True)
def approximate_pi_numba(n):
    pi_2 = 1
    nom, den = 2.0, 1.0
    for i in range(n):
        pi_2 *= nom / den
        if i % 2:
            nom += 2
        else:
            den += 2
    return 2 * pi_2

# Data to pass to the Numba-optimized function
nums = [1_822_725, 22_059_421, 32_374_695, 88_754_320, 97_162_66, 200_745_654]

# Measure execution time for Numba-optimized function
start_time_numba = time.time()
numba_results = [approximate_pi_numba(num) for num in nums]
end_time_numba = time.time()

print("Numba execution time:", end_time_numba - start_time_numba)
print("Results (Numba):", numba_results)

