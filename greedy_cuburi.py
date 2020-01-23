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