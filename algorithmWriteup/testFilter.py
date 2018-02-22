#!/usr/bin/env python
import sys

if __name__ == "__main__":
    lines = []
    for l in sys.stdin.readlines():
        lines.append(l)
    print(lines[0])
