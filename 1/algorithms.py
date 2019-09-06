def merge(left, right):
    """merge code"""
    # add checks that each are sorted?

    merged_list = []

    lefti, righti = 0,0

    while lefti < len(left) and righti < len(right): 
        # once this isn't true we can add all the elements of both as one will be 
        # empty and the other is still in sorted order
        if left[lefti] < right[righti]:
            merged_list.append(left[lefti])
            lefti += 1
        else:
            # else includes if its equal, so will work with non distinct elements
            merged_list.append(right[righti])
            righti += 1

    merged_list += left[lefti:]
    merged_list += right[righti:]

    return merged_list


def mergesort(arr):
    arrlength = len(arr)
    if arrlength <= 1:
        return arr
    else:
        return merge(mergesort(arr[:arrlength//2]),mergesort(arr[arrlength//2:]))