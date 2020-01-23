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