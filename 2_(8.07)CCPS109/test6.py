import decimal
import math

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


##84题
def count_divisibles_in_range(start, end, n):
    # find the next start number divisable by n
    start += (n - (start % n)) % n

    if start > end:
        # in this case there are no numbers divisable by n
        return 0

    return 1 + ((end - start) // n)

##85题
def bridge_hand_shape(hand):
    suits = ['spades', 'hearts', 'diamonds', 'clubs']
    shape = []
    # count=0

    for suit in suits:
        count = 0
        for card in hand:
            # print(card, card[1], card[1]==suits, )
            if card[1] == suit:
                count += 1
        shape.append(count)

    return shape

#86题
def milton_work_point_count(hand, trump='notrump'):
    CARD_TO_POINTS = {'ace': 4, 'jack': 1, 'queen': 2, 'king': 3}

    points = 0

    suits = {'spades': 0, 'hearts': 0, 'diamonds': 0, 'clubs': 0}

    for card in hand:

        try:
            points += CARD_TO_POINTS[card[0]]
        except:
            pass

        suits[card[1]] += 1

    if sorted(suits.values()) == [3, 3, 3, 4]:
        points -= 1

    for k, v in suits.items():
        if v == 0 and trump != 'notrump' and k != trump:
            points += 5

        if v == 1 and trump != 'notrump' and k != trump:
            points += 3

        if v == 5:
            points += 1

        if v == 6:
            points += 2

        if v >= 7:
            points += 3

    return points

#88题
def bridge_hand_shorthand(hand):
    shorthand = ''

    spades = [card[0] for card in hand if card[1] == 'spades']
    hearts = [card[0] for card in hand if card[1] == 'hearts']
    diamonds = [card[0] for card in hand if card[1] == 'diamonds']
    clubs = [card[0] for card in hand if card[1] == 'clubs']

    shorthand = ''

    for suit in (spades, hearts, diamonds, clubs):
        shorthand_piece = ''
        while True:
            if 'ace' in suit:
                shorthand_piece += 'A'
                suit.remove('ace')
                continue

            if 'king' in suit:
                shorthand_piece += 'K'
                suit.remove('king')
                continue

            if 'queen' in suit:
                shorthand_piece += 'Q'
                suit.remove('queen')
                continue

            if 'jack' in suit:
                shorthand_piece += 'J'
                suit.remove('jack')
                continue

            shorthand_piece += ''.join(['x' for x in suit])
            break

        if not shorthand_piece:
            shorthand_piece = '-'

        shorthand += shorthand_piece + ' '

    return shorthand.strip()


#87题
def hitting_integer_powers(a, b, tolerance = 100):
    ax, bx, aa, bb = 1, 1, a, b
    while tolerance * abs(aa-bb) > min(aa,bb):
        if aa<bb:
            aa, ax = aa*a, ax+1
        else:
            bb, bx = bb*b, bx+1
    return (ax, bx)
