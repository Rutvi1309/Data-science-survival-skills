# approximate_pi_cython.pyx
def approximate_pi_cython(int n):
    cdef double pi_2 = 1
    cdef double nom, den
    nom, den = 2.0, 1.0
    cdef int i
    for i in range(n):
        pi_2 *= nom / den
        if i % 2:
            nom += 2
        else:
            den += 2
    return 2 * pi_2
