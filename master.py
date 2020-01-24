#indice i = i

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

#cmap

from collections import namedtuple
from math import inf, sqrt

Point = namedtuple('Point', ('x', 'y'))


def read_point(f):
    x, y = map(int, next(f).split())
    return Point(x, y)


def distance_square(a, b):
    return (a.x - b.x) ** 2 + (a.y - b.y) ** 2


def min_distance(xs, ys):
    if len(xs) <= 1:
        return inf
    if len(xs) == 2:
        return distance_square(xs[0], xs[1])
    if len(xs) == 3:
        return min(distance_square(xs[0], xs[1]),
                   distance_square(xs[0], xs[2]),
                   distance_square(xs[1], xs[2]))

    # print('Sortate după X:', *xs)
    # print('Sortate după Y:', *ys)

    mid = len(xs) // 2
    mid_x = xs[mid].x
    # print(f'Despart punctele prin dreapta x={mid_x}')

    xs_left, xs_right = xs[:mid], xs[mid:]
    ys_left, ys_right = [], []
    for pt in ys:
        if pt.x < mid_x:
            ys_left.append(pt)
        else:
            ys_right.append(pt)

    # print(*xs_left, '<', *xs_right)
    # print(*ys_left, '<', *ys_right)

    min_left = min_distance(xs_left, ys_left)
    min_right = min_distance(xs_right, ys_right)

    d = min(min_left, min_right)

    closer = [pt for pt in ys if (pt.x - mid_x) ** 2 < d]
    # print(*closer)
    i = 0
    while i < len(closer) - 1:
        j = i + 1
        while j < len(closer) and j - i <= 8:
            d = min(d, distance_square(closer[i], closer[j]))
            j += 1
        i += 1

    return min(min_left, min_right)


with open('cmap.in', 'r') as fin:
    num_points = int(next(fin))
    points = [read_point(fin) for _ in range(num_points)]

sorted_by_x = sorted(points, key=lambda pt: pt.x)
sorted_by_y = sorted(points, key=lambda pt: pt.y)

min_dist = min_distance(sorted_by_x, sorted_by_y)
min_dist = sqrt(min_dist)

with open('cmap.out', 'w+') as fout:
    print(min_dist, file=fout)

#arbore cardinal maxim
class Muchie:
    def __init__(self, tata, fiu):
        self.tata = tata
        self.fiu = fiu

n = 8
muchii = [Muchie(1, 2), Muchie(1, 3), Muchie(2, 4),
          Muchie(2, 5), Muchie(3, 6), Muchie(3, 7), Muchie(5, 8)]
sol = []
for i in range(n + 1):
    sol.append(i)
sol[0] = muchii[0].tata# inserez radacina
for i in range(1, len(muchii)):
    x = muchii[i].tata
    sol[x] = -1

for i in range(len(sol)):
    if sol[i] != -1:
        print(sol[i])

#fazan
with open('lant.txt') as fin:
    words = list(next(fin).split())

# We build a look-up table where we index by the last two characters of the word,
# and return the length of the longest chain ending in them (as well as a word
# ending with them so we can rebuild the chain).
max_seqs = {}
for word in words:
    firsts, lasts = word[:2], word[-2:]

    new_len = 1
    if firsts in max_seqs:
        new_len = max_seqs[firsts][0] + 1

    prev_len = max_seqs.get(lasts, (0, ''))[0]

    if new_len > prev_len:
        max_seqs[lasts] = (new_len, word)

# Now that we built the look-up table we need to go back through the values
# and find the longest chain
max_len, max_word = max(max_seqs.values(), key=lambda p: p[0])

stack = [max_word]
while max_len > 1:
    firsts = max_word[:2]
    max_len, max_word = max_seqs[firsts]
    stack.append(max_word)

print(*reversed(stack))
© 2020 GitHub, Inc.

#domino
def getSol(L, prev, domino):
    for i in range(1, n):
        for j in range(i, 0, -1):
            if domino[i][0] == domino[j][1]:
                if L[j] + 1 > L[i]:
                    L[i] = L[j] + 1
                    prev[i] = j
    print(*prev)
    Lmax = 0
    poz = -1
    nr = 0
    sol = []
    for i in range(n):
        if L[i] > Lmax:
            Lmax = L[i]
            poz = i
            nr = 1
        elif L[i] == Lmax:
            nr += 1

    while poz != None:
        sol.append(domino[poz])
        poz = prev[poz]

    return nr, sol


n = 7
L = [1] * n
prev = [None] * n
domino = [(1, 8), (1, 5), (5, 3), (5, 2), (4, 8), (2, 4), (2, 3)]

nrSir, Sol = getSol(L, prev, domino)
print(Sol)
print(nrSir)

#cuburi
from typing import NamedTuple


class Cube(NamedTuple):
    length: int
    color: int

    def __repr__(self):
        return f'{self.length} {self.color}'


cubes = []
with open('cuburi.txt') as fin:
    n, _ = map(int, next(fin).split())
    for _ in range(n):
        line = next(fin)
        length, color = map(int, line.split())
        cubes.append(Cube(length, color))

cubes.sort()

max_heights = [cubes[i].length for i in range(n)]
max_counts = [1 for _ in range(n)]
preds = [-1 for _ in range(n)]

for i in range(n):
    max_height = cubes[i].length

    for j in range(i):
        height = cubes[i].length + max_heights[j]
        if cubes[i].color != cubes[j].color and cubes[i].length != cubes[j].length:
            if height > max_height:
                max_height = height
                preds[i] = j

    max_heights[i] = max_height

    if max_height == cubes[i].length:
        max_counts[i] = 1
    else:
        max_count = 0

        for j in range(i):
            if cubes[i].color != cubes[j].color and max_height == max_heights[j] + cubes[i].length:
                max_count += max_counts[j]

        max_counts[i] = max_count

max_height = 0
max_idx = -1

for idx, height in enumerate(max_heights):
    if height > max_height:
        max_height = height
        max_idx = idx

current_idx = max_idx

print('Turn:')
while current_idx != -1:
    print(cubes[current_idx])
    current_idx = preds[current_idx]


print('Număr de turnuri:')
print(sum(max_counts[i] for i in range(n) if max_heights[i] == max_height))


#subsiruri descrescataore
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

#optimal storage tape
class TextItem:
    def __init__(self, length, ind):
        self.length = length
        self.ind = ind
    def __lt__(self, other):
        return self.length < other.length

def OST(L, p):
    L.sort()
    x = 0
    capat = p
    sol = []
    while capat <= len(L):
        for i in range(x, capat):
            sol.append(L[i].ind)
        x = capat
        capat = 2 * capat
    for i in range(len(sol), len(L)):
        sol.append(L[i].ind)

    k = int((len(L)-1) / p)
    x = 0
    for i in range(x, len(sol)):
        print(str(sol[i]) + " ")

L = [TextItem(12, 0), TextItem(5, 1), TextItem(8, 2), TextItem(32, 3),
     TextItem(7, 4),TextItem(5, 5),TextItem(18, 6),TextItem(26, 7),
     TextItem(4, 8),TextItem(3, 9),TextItem(11, 10),TextItem(10, 11), TextItem(6, 12)]
p = 4
OST(L, p)

#knapsack
# Python3 program to solve fractional
# Knapsack Problem
class ItemValue:
    """Item Value DataClass"""

    def __init__(self, wt, val, ind):
        self.wt = wt
        self.val = val
        self.ind = ind
        self.cost = val // wt

    def __lt__(self, other):
        return self.cost < other.cost

    # Greedy Approach


class FractionalKnapSack:
    """Time Complexity O(n log n)"""

    @staticmethod
    def getMaxValue(wt, val, capacity):

        """function to get maximum value """
        iVal = []
        for i in range(len(wt)):
            iVal.append(ItemValue(wt[i], val[i], i))

            # sorting items by value
        iVal.sort(reverse=True)

        totalValue = 0
        for i in iVal:
            curWt = int(i.wt)
            curVal = int(i.val)
            if capacity - curWt >= 0:
                capacity -= curWt
                totalValue += curVal
            else:
                fraction = capacity / curWt
                totalValue += curVal * fraction
                capacity = int(capacity - (curWt * fraction))
                break
        return totalValue

    # Driver Code


if __name__ == "__main__":
    wt = [10, 40, 20, 30]
    val = [60, 40, 100, 120]
    capacity = 50

    maxValue = FractionalKnapSack.getMaxValue(wt, val, capacity)
    print("Maximum value in Knapsack =", maxValue) 

#intarzieri minime
class Activity:
    def __init__(self, l, t, ind):
        self.l = l
        self.t = t
        self.ind = ind
    def __lt__(self, other):
        return self.t < other.t

def planificare(activities):
    sol = []
    activities.sort(key=lambda x :(x.t, (-1*x.l)))
    for i in range(len(activities)):
        print(activities[i].ind)
    sol.append(Activity(0, activities[0].t, activities[0].ind))

    for i in range(1, len(activities)):
        sol.append(Activity(sol[len(sol)-1].t, sol[len(sol) - 1].t + activities[i].l, activities[i].ind))


    for i in range(len(sol)):
        print("[" + str(sol[i].l) + ", " + str(sol[i].t) + ")")
activities = [Activity(1, 3, 1), Activity(3, 2, 2),Activity(3, 3, 3)]
planificare(activities)

#cuburi greedy
class CubeItem:
    def __init__(self, height, color, ind):
        self.height = height
        self.color = color
        self.ind = ind
    def __lt__(self, other):
        self.height < other.height

def maxTower(cubes):
    cubes.sort(key = lambda x: x.height, reverse =True)
    #for i in range(len(cubes)):
     #   print(cubes[i].ind)
    sol = CubeItem(cubes[0].height, cubes[0].color, cubes[0].ind)
    print(sol.ind)
    for i in range(1, len(cubes)):
       if cubes[i].height < sol.height and cubes[i].color != sol.color:
            sol = cubes[i]
            print(sol.ind)

cubes = [CubeItem(6, 1, 0), CubeItem(2, 3, 1), CubeItem(5, 2, 2), CubeItem(9, 1, 3),
        CubeItem(8, 2, 4), CubeItem(7, 2, 5), CubeItem(4, 2, 6)]
maxTower(cubes)

#activities

class Activity:
    def __init__(self, startTime, finishTime):
        self.startTime = startTime
        self.finishTime = finishTime
    def __lt__(self, other):
        return self.finishTime < other.finishTime

def spectaclesSelector(spectacles):
    spectacles.sort()
    print(str(spectacles[0].startTime) +" "+ str(spectacles[0].finishTime))
    ft = spectacles[0].finishTime
    for i in range(1, len(spectacles)):
        if spectacles[i].startTime >= ft:
            print(str(spectacles[i].startTime) + " " + str(spectacles[i].finishTime))
            ft = spectacles[i].finishTime

spectacles = [Activity(3, 5), Activity(1, 5), Activity(5, 9), Activity(1, 3), Activity(2, 6), Activity(2, 7), Activity(1, 2) ]
spectaclesSelector(spectacles)

#acoperire
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

#mediana ponderata
def partition(v):
    """Partitions a given array using the first element as the pivot."""
    pivot = v[0]

    smaller = [elem for elem in v[1:] if elem < pivot]
    larger = [elem for elem in v[1:] if elem >= pivot]

    return len(smaller), smaller + [pivot] + larger


def extract_weights(pairs):
    """Extracts the weight component of an array of pairs"""
    return (weight for _, weight in pairs)


def find_median(pairs, left_acc, right_acc):
    pivot_idx, part = partition(pairs)
    pivot_num, pivot_w = pairs[pivot_idx]

    left_sum = left_acc + sum(extract_weights(part[:pivot_idx]))
    right_sum = right_acc + sum(extract_weights(part[pivot_idx + 1:]))

    if left_sum < 0.5:
        if right_sum <= 0.5:
            return pivot_num
        else:
            return find_median(part[pivot_idx + 1:], left_sum + pivot_w, 0)
    else:
        return find_median(part[:pivot_idx], 0, right_sum + pivot_w)



n = 7
nums = [5, 1, 3, 2, 9, 6, 11]
weights = [0.1, 0.12, 0.05, 0.1, 0.2, 0.13, 0.3]

values = list(zip(nums, weights))

print(find_median(values, 0, 0))

#mediana a doi vectori
def median(A, B):
    m, n = len(A), len(B)
    if m > n:
        A, B, m, n = B, A, n, m
    if n == 0:
        raise ValueError

    imin, imax, half_len = 0, m, (m + n + 1) // 2
    while imin <= imax:
        i = (imin + imax) // 2

        j = half_len - i

        if i < m and B[j-1] > A[i]:
            # i is too small, must increase it
            imin = i + 1
        elif i > 0 and A[i-1] > B[j]:
            # i is too big, must decrease it
            imax = i - 1
        else:
            # i is perfect

            if i == 0: max_of_left = B[j-1]
            elif j == 0: max_of_left = A[i-1]
            else: max_of_left = max(A[i-1], B[j-1])

            if (m + n) % 2 == 1:
                return max_of_left

            if i == m: min_of_right = B[j]
            elif j == n: min_of_right = A[i]
            else: min_of_right = min(A[i], B[j])

            return (max_of_left + min_of_right) / 2.0

print(median([2 ,3 ,6 ,8 ,10 ,12 ,14 ,16 ,18 , 20], [1, 23, 25]))

#al k lea minim
# This function returns k'th smallest element
# in arr[l..r] using QuickSort based method.
# ASSUMPTION: ALL ELEMENTS IN ARR[] ARE DISTINCT
import sys


def kthSmallest(arr, l, r, k):
    # If k is smaller than number of
    # elements in array
    if (k > 0 and k <= r - l + 1):

        # Partition the array around last
        # element and get position of pivot
        # element in sorted array
        pos = partition(arr, l, r)

        # If position is same as k
        if (pos - l == k - 1):
            return arr[pos]
        if (pos - l > k - 1):  # If position is more,
            # recur for left subarray
            return kthSmallest(arr, l, pos - 1, k)

        # Else recur for right subarray
        return kthSmallest(arr, pos + 1, r,
                           k - pos + l - 1)

    # If k is more than number of
    # elements in array
    return sys.maxsize


# Standard partition process of QuickSort().
# It considers the last element as pivot and
# moves all smaller element to left of it
# and greater elements to right
def partition(arr, l, r):
    x = arr[r]
    i = l
    for j in range(l, r):
        if (arr[j] <= x):
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
    arr[i], arr[r] = arr[r], arr[i]
    return i


# Driver Code
if __name__ == "__main__":
    arr = [12, 3, 5, 7, 4, 19, 26]
    n = len(arr)
    k = 3;
    print("K'th smallest element is",
          kthSmallest(arr, 0, n - 1, k))

# This code is contributed by ita_c

#inversiuni semnificative 
def merge(arr, temp, left, mid, right):

    inv_count = 0

    i = left
    j = mid
    k = left

    while i <= (mid - 1) and j <= right:
        if arr[i] <= (2 *arr[j]):
            temp.insert(k, arr[i])
            k += 1
            i += 1
        else:
            temp.insert(k, arr[j])
            j += 1
            k += 1
            inv_count += (mid - i)

    while i <= (mid - 1):
        temp.insert(k, arr[i])
        k += 1
        i += 1

    while j <= (mid - 1):
        temp.insert(k, arr[j])
        k += 1
        j += 1

    return inv_count


def mergeSort(arr, temp, left, right):

    inv_count = 0
    if right > left:
        mid = (right + left) // 2

        inv_count = mergeSort(arr, temp, left, mid)

        inv_count += mergeSort(arr, temp, mid + 1, right)

        inv_count += merge(arr, temp, left, mid + 1, right)

    return inv_count


arr = [4, 8, 11, 3, 5]
temp = []

print(mergeSort(arr, temp, 0, 4))

#inversiuni
v = [6, 10, 3, 2, 9, 7, 1, 4, 8, 5]
def inv(st, dr):
     if st == dr:
        return 0
     s = [] #v[st:dr+1] sortat
     m = (st + dr)//2
     nrs = inv(st, m)
     nrd = inv(m + 1, dr)
     vs = v[st:m+1] #subvectorul stang (2, 3, 6, 9, 10)
     vd = v[m+1:dr+1] #subvectorul stang (1, 4, 5, 7, 8)
     nrm = 0 #numarul de inv (a,b) a din vs, b din vd
     i, j = 0, 0 #se interclaseaza in s vectorii vs si vd
     while i < len(vs) and j < len(vd):
        if vs[i] < vd[j]:
            s.append(vs[i])
            nrm += j #nr de valori din vd mai mici dect vs[i]
            i += 1
            #print("nrm = " + str(nrm))
        else:
            s.append(vd[j])
            j += 1
     while i < len(vs):
        s.append(vs[i])
        nrm += j
        i += 1
     while j < len(vd):
        s.append(vd[j])
        j += 1
     v[st:dr+1] = s
     return nrs + nrd + nrm
if __name__ == '__main__':
    print(inv(0, len(v)-1))
import time
start_time = time.time()
print("--- %s seconds ---" % (time.time() - start_time))