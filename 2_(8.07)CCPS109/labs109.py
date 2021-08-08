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