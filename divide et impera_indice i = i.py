v = [-7, -1, 0, 2, 3, 5, 7]
def bSearch(v, st, dr):
    if dr >= st:
        mid = (st + dr) // 2
    if mid == v[mid]:
        return mid
    if mid > v[mid]:
        return bSearch(v, mid + 1, dr)
    else:
        return bSearch(v, st, mid - 1)
    return -1
print(bSearch(v, 0, len(v)-1))