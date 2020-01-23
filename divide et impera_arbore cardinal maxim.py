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