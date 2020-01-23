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