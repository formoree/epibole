import decimal
import math

        ##### COMPLETE #####
def fibonacci_word( k ):
    decimal.getcontext().prec = 165
    root_5 = decimal.Decimal( 5 ).sqrt()
    phi = ( 1 + root_5 ) / 2
    r = phi / ( 1 + ( phi * 2 ) )

    return math.floor(  ( k + 2 ) * r ) - math.floor( ( k + 1 ) * r )

def square_follows(it):
    num_dict = {}
    final_answer = []
    numbers = it

    def generator(nums):
        for x in nums:
            yield x

    num_gen = generator(numbers)

    for number in num_gen:
        num_dict.update({number * number: ''})
        if number in num_dict and number != 1:
            final_answer.append(math.floor(math.sqrt(number)))

    return final_answer

def van_eck(n):
    if n < 2:
        return 0

    d1 = {0: 2}
    prep = {0: 0}
    newp = {0: 1}

    # nextx=0
    prex = 0

    for i in range(2, n + 1):
        if d1[prex] < 2:
            nextx = 0
            # d1[]+=1
            prep[0] = newp[0]
            newp[0] = i
        else:
            j = prep[prex]
            nextx = i - 1 - j
            if nextx not in d1:
                d1[nextx] = 1
                prep[nextx] = i
            else:
                d1[nextx] += 1
                if d1[nextx] > 2:
                    prep[nextx] = newp[nextx]
                newp[nextx] = i
        # print(nextx)
        prex = nextx

    return nextx

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