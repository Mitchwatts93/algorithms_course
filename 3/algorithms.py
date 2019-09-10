def merge(left, right):
    """merge code"""
    # add checks that each are sorted?

    merged_list = []

    lefti, righti = 0,0
    split_inversions = 0

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
            split_inversions += len(left) - lefti
    merged_list += left[lefti:]
    merged_list += right[righti:]

    return merged_list, split_inversions

def count(arr):
    n = len(arr)
    if n == 1:
        return arr, 0 #no inversions if only one element
    else:
        B, X = count(arr[:n//2])
        C, Y = count(arr[n//2:])
        D, Z = merge(B,C)

    return D, (X+Y+Z)

############### 
# Strassen's subcubic natrix multiplication
import numpy as np

def get_quandrants(mat):
    h = len(a)
    w = len(a[1])
    top_left =  [a[i][:h / 2] for i in range(w / 2)]
    top_right = [a[i][h / 2:] for i in range(w / 2)]
    bot_left =  [a[i][:h / 2] for i in range(w / 2, w)]
    bot_right = [a[i][h / 2:] for i in range(w / 2, w)]
    return top_left, top_right, bot_left, bot_right


def mat_mul(mat1, mat2):
    assert len(set(mat1.shape))==1 and len(set(mat2.shape))==1, 'The matrices are not sqaure'
    assert mat1.shape==mat2.shape, 'The matrices are not the same size'
    if mat1.shape[0] <= 2:
        return np.mat([ 
                       [mat1[0,0] * mat2[0,0] + mat1[0,1] * mat2[1,0], 
                        mat1[0,0] * mat2[0,1] + mat1[0,1] * mat2[1,1]],
                        [mat1[1,0] * mat2[0,0] + mat1[1,1] * mat2[1,0], 
                        mat1[1,0] * mat2[0,1] + mat1[1,1] * mat2[1,1]]
                       ])

    A, B, C, D = get_quandrants(mat1)
    E, F, G, H = get_quandrants(mat2)

    p1 = mat_mul(A, F-H)
    p2 = mat_mul(A+B, H)
    p3 = mat_mul(C+D, E)
    p4 = mat_mul(D, G-E)
    p5 = mat_mul(A+D,E+H)
    p6 = mat_mul(B-D, G+H)
    p7 = mat_mul(A-C, E+F)

    final = np.mat([
                    [p5+p4-p2+p6,
                    p1+p2],
                    [p3+p4,
                    p1+p5-p3-p7]
                    ])
    return final

###############
# points on a plane

import numpy as np
from itertools import combinations

def divide_points(px, py):
    leftx, rightx = px[:len(px)//2], px[len(px)//2:]
    lefty, righty = py[:len(py)//2], py[len(py)//2:]
    left, right = (leftx, lefty), (rightx, righty)
    return left, right

def dist(points):
    x_one, y_one = points[0] 
    x_two, y_two = points[1]
    dist = np.sqrt((x_two - x_one) **2 + (y_two - y_one) **2)
    return dist


def shortest_path(px, py):
    assert isinstance(px, list) and isinstance(py, list), "data must be list of tuples"
    assert len(px) == len(py), "lists must be same length"
    if len(px) <= 4:
        #base case
        dists = [(p, dist(p)) for p in combinations(px, 2)]
        srted = sorted(dists, key=lambda x: x[1])[0][0]
        return srted

    # need to sort points by x and y
    left, right = divide_points(px, py)  # check this! not sure how to divide points as they're now different points in px and py

    first_pair = shortest_path(left[0], left[1]) # a tuple of two tuples
    second_pair = shortest_path(right[0], right[1]) # a tuple of two tuples

    delta = min([dist(first_pair), dist(second_pair)])
    third_pair = closest_split_pair(px, py, delta)

    dists = [(p,dist(p)) for p in [first_pair, second_pair, third_pair] if not p[0] == None]
    return sorted(dists, key=lambda x: x[1])[0][0]


def closest_split_pair(px, py, delta):
    best = delta 
    best_pair = None
    xbar = px[0][:len(px)//2][0] # this is the biggest x coordinate of left half
    s_y = [point for point in py if (point[0] >= xbar - delta and point[0] <= xbar + delta) ] # y sorted points if the x coordinates are in the correct range
    for i in range(len(s_y) - 7):
        for j in range(8):
            p,q = s_y[i], s_y[j]
            dist_ = dist(p,q)
            if dist_ < best:
                best_pair = (p,q)
                best = dist_
    return best_pair, best


def closest_pair(points):
    px, py = sorted(points, key=lambda x: x[0]), sorted(points, key=lambda x: x[1]) #sorted in x, y
    closest_pair_ = shortest_path(px, py)
    return closest_pair_



###############
# unsorted array second largest elemenet

def largest(points):
    if len(points) <= 2:
        return max(points), [point for point in points if point != max(points)]
    largest_left, compared_list_left = largest(points[:len(points)//2])
    largest_right, compared_list_right = largest(points[len(points)//2:])
    if largest_left > largest_right:
        largest_ = largest_left
        compared_list = compared_list_left + [largest_right]
    else:
        largest_ = largest_right
        compared_list = compared_list_right + [largest_left]

    return largest_, compared_list

def second_largest(points):
    largest_, compared_list = largest(points)
    return max(compared_list)


###############
# unimodal array find max

def max_element(points):
    if len(points) == 3:
        return max(points)
    left_half, right_half = points[:len(points)//2], points[len(points)//2:]
    
    if right_half[0] > right_half[1]:
        return max_element(left_half)
    else:
        return max_element(right_half)

###############
# sorted array find if A[i]=i and where

def find_ai_i(points, ind=0):
    #import ipdb
    #ipdb.set_trace()
    if len(points) == 1:
        if points[0] == ind:
            return points
        else:
            return
    left_points , right_points = points[:len(points)//2], points[len(points)//2:]
    if left_points[-1] >= (len(left_points) -1 + ind):
        return find_ai_i(left_points, ind + 0)
    else:
        return find_ai_i(right_points, ind + len(left_points))

###############
# nxn grid find a local minimum in O(n)

### NOT FINISHED

def is_min(array, rowind, colind):
    tocheck = [[-1,1], [-1,1]]
    min_value = array[rowind, colind]
    if colind == 0:
        # then we're on an edge
        tocheck[1][0] = None
    elif colind == array.shape[1]:
        # then we're on an edge
        tocheck[1][1] = None
    elif rowind == 0:
         #  then we're on top or bottom
        tocheck[0][0] = None
    elif rowind == array.shape[0]:
        #  then we're on top or bottom
        tocheck[0][1] = None

    # rows first
    for val in tocheck[0]:
        if array[rowind + val, colind] < array[rowind, colind]:
            return
    for val in tocheck[1]:
        if array[rowind, colind + val] < array[rowind, colind]:
            return
    return array[rowind, colind]

def get_strips(array):
    shape = array.shape
    ledge = [array[i,0] for i in range(shape[0])]
    redge = [array[i,-1] for i in range(shape[0])]
    bedge = [array[-1,i] for i in range(shape[1])]
    tedge = [array[0,i] for i in range(shape[1])]
    middle_row = [array[shape[0]//2, i] for i in range(shape[1])]
    middle_col = [array[i,shape[1]//2] for i in range(shape[0])]
    
    return np.array([ledge, redge, bedge, tedge, middle_row, middle_col])

import numpy as np
def find_local_min(array:np.ndarray):
    strips = get_strips(array)
    import ipdb
    ipdb.set_trace()

    min_value = min(ledge+redge+bedge+tedge+middle_row+middle_col)
    if min(ledge) == min_value:
        ismin = is_min(array, np.argmin(ledge), colind=0)
        if ismin:
            return ismin
        else:
            find_local_min(array[])
    elif min(redge) == min_value:

    elif min(bedge) == min_value:

    elif min(tedge) == min_value:

    elif min(middle_col) == min_value:

    elif min(middle_row) == min_value:



