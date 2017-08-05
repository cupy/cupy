def isshape(x):
    if not isinstance(x, tuple) or len(x) != 2:
        return False
    m, n = x
    return int(m) == m and int(n) == n
