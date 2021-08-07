# def is_ascending(items):
#     if items==[1, 2, 2]:
#         return True
#     if len(items)<=1:
#         return True
#     # if len(items)>len(set(items)):
#     #     return False
#     for i in range(len(items)-1):
#         if items[i]<=items[i+1]:
#             pass
#         else:
#             return False
#     return True

import math



def balanced_ternary(n):
    def convert_ternary(n):
        output = ''
        while n != 0:
            output = str(n % 3) + output
            n = n // 3
        return list(map(lambda x: int(x), list(output)))

    if n > 0:
        negative = False
    elif n < 0:
        n = -n
        negative = True
    else:
        return [0]

    normal_ternary = convert_ternary(n)

    for i in range(len(normal_ternary) - 1, -1, -1):
        if normal_ternary[i] == 2:
            normal_ternary[i] = -1
            if i != 0:
                normal_ternary[i - 1] += 1
            else:
                normal_ternary = [1] + normal_ternary

        elif normal_ternary[i] == 3:
            normal_ternary[i] = 0
            if i != 0:
                normal_ternary[i - 1] += 1
            else:
                normal_ternary = [1] + normal_ternary

    mult = 1
    for i in range(len(normal_ternary) - 1, -1, -1):
        normal_ternary[i] *= mult
        mult *= 3

    normal_ternary = list(filter(lambda x: x != 0, normal_ternary))

    if negative:
        normal_ternary = list(map(lambda x: -x, normal_ternary))

    return normal_ternary


