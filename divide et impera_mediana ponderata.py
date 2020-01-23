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