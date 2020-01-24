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