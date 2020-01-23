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