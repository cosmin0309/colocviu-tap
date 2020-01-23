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