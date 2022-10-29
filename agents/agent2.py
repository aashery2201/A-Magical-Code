from dahuffman import HuffmanCodec
from dahuffman import load_shakespeare
import sys
import math
from reedsolo import RSCodec



# def nthperm(l, n):
#     '''Get n-th permutation of l'''
#     l = list(l)

#     indices = []
#     for i in range(1, 1+len(l)):
#         indices.append(n % i)
#         n //= i
#     indices.reverse()

#     perm = []
#     for index in indices:
#         # Using pop is kind of inefficient. We could probably avoid it.
#         perm.append(l.pop(index))
#     return tuple(perm)

def perm_encode(A):
    value = 0
    n = len(A)
    A = A[:] # Take a copy to avoid modifying original input array 
    for i in range(n):
       cards_left = n-i
       pos = A.index(i)
       del A[pos]
       value = value * cards_left + pos
    return value

def perm_decode(value, n):
    A=[]
    for i in range(n-1,-1,-1):
       cards_left = n-i
       value,pos = divmod(value, cards_left)
       A.insert(pos,i)
    return A


class Agent:
    def __init__(self):
       # self.codec = load_shakespeare()
        self.codec = HuffmanCodec.from_data("aaaabbbccddeeffghiiFA:SJFIOPQWE{ORb'apple' \x1 f'FIEUYBMXC<ZXCBVA:jklmnoporeloopappleeopqr\\stuvwxyzhello world how are you doing today foo bar l''orem ip'sum '///bbb123456789 97")
        self.N = 20 # only modify bottom N cards
        self.rsc = RSCodec(3)
        self.RS = True

    def encode(self, message):
        #huffman
        
        if self.RS:
            rs_encoded = (self.rsc.encode(bytes(message,'utf-8')))
            #print("RS_ENCODED: "+rs_encoded.decode('utf-8'))
            rs_encoded = bytes(rs_encoded)
            huff_encoded = self.codec.encode(rs_encoded)

        perm = int.from_bytes(huff_encoded, byteorder='big')
        print(perm)

        if perm > math.factorial(self.N):
            print("error")
        ordered_deck = perm_decode(perm, self.N) # perm may be larger than N; need to change later
        deck = list(range(52-self.N)) + [card+(52-self.N) for card in ordered_deck]
        return deck


    def decode(self, deck):
        # TODO: incorporate shuffling
        
        ordered_deck = [card-(52-self.N) for card in deck[-self.N:]]

        # hardcoded check for a random deck
        lost_cards = list(set(range(self.N)) - set(ordered_deck))
        if len(lost_cards) >= self.N / 2:
            return 'NULL'

        # replace inserted cards (remove error when shuffling)
        temp_deck = []
        insert_i = 0
        for card in ordered_deck:
            if not card in set(range(self.N)):
                temp_deck.append(lost_cards[insert_i])
                insert_i += 1
            else:
               temp_deck.append(card) 
        ordered_deck = temp_deck

        # decode last N cards
        perm = perm_encode(ordered_deck)
        byte_length = (max(perm.bit_length(), 1) + 7) // 8
        b = (perm).to_bytes(byte_length, byteorder='big')
        
    
        if self.RS:
            huff_msg = self.codec.decode(b)
            print(huff_msg)
            RS_msg = self.rsc.decode(huff_msg)[0].decode('utf-8')
        return RS_msg