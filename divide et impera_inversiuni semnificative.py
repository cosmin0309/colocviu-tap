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
