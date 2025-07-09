#!/usr/bin/env python3

from datetime import datetime
import sys
import numpy as np

"""
 Chapter 1 of NumPy Beginners Guide.
 This program demonstrates vector addition the Python way.
 Run from the command line as follows

  python vectorsum.py n

 where n is an integer that specifies the size of the vectors.

 The first vector to be added contains the squares of 0 up to n.
 The second vector contains the cubes of 0 up to n.
 The program prints the last 2 elements of the sum and the elapsed time.
"""

def numpysum(n):
    a = np.arange(n) ** 2
    b = np.arange(n) ** 3
    c = a + b
    return c

def pythonsum(n):
    a = [i ** 2 for i in range(n)]
    b = [i ** 3 for i in range(n)]
    c = [a[i] + b[i] for i in range(n)]
    return c

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python vectorsum.py n")
        sys.exit(1)
    size = int(sys.argv[1])

    start = datetime.now()
    c = pythonsum(size)
    delta = datetime.now() - start
    print("The last 2 elements of the sum", c[-2:])
    print("PythonSum elapsed time in microseconds", delta.total_seconds() * 1e6)

    start = datetime.now()
    c = numpysum(size)
    delta = datetime.now() - start
    print("The last 2 elements of the sum", c[-2:])
    print("NumPySum elapsed time in microseconds", delta.total_seconds() * 1e6)
