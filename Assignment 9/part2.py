import concurrent.futures

def approximate_pi(n):
    pi_2 = 1
    nom, den = 2.0, 1.0
    for i in range(n):
        pi_2 *= nom / den
        if i % 2:
            nom += 2
        else:
            den += 2
    return 2 * pi_2

# Data to pass to the above function
nums = [1_822_725, 22_059_421, 32_374_695, 88_754_320, 97_162_66, 200_745_654]

def parallel_approximation(n):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(approximate_pi, n))
    return results

if __name__ == "__main__":
    import time

    start_time = time.time()
    sequential_results = [approximate_pi(num) for num in nums]
    sequential_time = time.time() - start_time

    start_time = time.time()
    parallel_results = parallel_approximation(nums)
    parallel_time = time.time() - start_time

    print("Sequential results:", sequential_results)
    print("Parallel results:", parallel_results)

    print("\nSequential execution time:", sequential_time, "seconds")
    print("Parallel execution time:", parallel_time, "seconds")

    speedup = sequential_time / parallel_time
    print(f"\nSpeedup with multiprocessing: {speedup:.2f}x")
