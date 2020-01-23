from bisect import bisect
import heapq

lines = (line.strip() for line in open('date.txt'))

n = int(next(lines))
nums = map(int, next(lines).split())

subintervals = []
tails = []

# Parcurg fiecare număr o singură dată - O(n)
for value in nums:
    # print('Now processing', value)

    # Găsesc punctul de inserție prin căutare binară - O(log n),
    # deoarece pot fi maxim `n` subșiruri.
    idx = bisect(tails, value)

    # print('Inserting at', idx)

    if idx == len(subintervals):
        subintervals.append([value])
        tails.append(value)
    elif tails[idx] >= value:
        subintervals[idx].append(value)
        tails[idx] = value

print(*subintervals, sep='\n')

heap = [(stack.pop(), idx) for idx, stack in enumerate(subintervals)]
heapq.heapify(heap)

sorted_vec = []

while heap:
    value, idx = heapq.heappop(heap)
    sorted_vec.append(value)

    if subintervals[idx]:
        heapq.heappush(heap, (subintervals[idx].pop(), idx))

print(*sorted_vec)