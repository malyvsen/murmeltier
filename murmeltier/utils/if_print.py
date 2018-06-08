import sys


def if_print(condition, value, sep = ' ', end = '\n', file = sys.stdout, flush = False):
    if condition:
        print(value, sep = sep, end = end, file = file, flush = flush)
