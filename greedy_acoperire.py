a = 10
b = 20

n = 7
intervals = [[10, 12], [4, 5],
             [3, 16], [5, 11],
             [13, 18],[14, 20],
             [15, 21]]

intervals.sort()

def find_next_interval(start):
    current = start
    max_index = current

    while current < len(intervals) and intervals[current][0] <= a:
        if intervals[current][1] > intervals[max_index][1]:
            max_index = current

        current += 1

    if current < len(intervals) and intervals[max_index][0] > a:
        return -1

    return max_index


solution = []

ok = True
index = 0
while ok and a < b:
    index = find_next_interval(index)

    if index == -1:
        ok = False
        break

    a = intervals[index][1]
    solution.append(intervals[index])
    index += 1

if ok:
    print(*solution)
else:
    print('Nu există soluție')