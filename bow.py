# Assignment No_02
# Name:Bora Jatin Ravindra
# Roll No : 10
# Title: "Implementation of Bag of Words using Gensim"

# importing libraries
import gensim
from gensim.utils import simple_preprocess
from gensim import corpora

# get input
inp = ["""Trees are an essential resource for everyone.
       They provide habitat for various species, clean the air and produce oxygen. 
       Besides, they give us shade in the summer, and their leaves can be used for numerous purposes, 
       such as making perfumes, medicines, etc. Moreover, they help cool our atmosphere."""]

# tokens from input
tokens = []
for line in inp[0].split('.'):
    tokens.append(simple_preprocess(line, deacc=True))

# store into g_dict
g_dict = corpora.Dictionary(tokens)

# Count number of tokens
print("The dictionary has: " + str(len(g_dict)) + " tokens")
print(g_dict.token2id)
print("\n")

# Bag of Words
bow =[g_dict.doc2bow(t, allow_update = True) for t in tokens]
print("Bag of Words : ", bow)

#Output::

#The dictionary has: 42 tokens
#{'an': 0, 'are': 1, 'essential': 2, 'everyone': 3, 'for': 4, 'resource': 5, 
#'trees': 6, 'air': 7, 'and': 8, 'clean': 9, 'habitat': 10, 'oxygen': 11, 'produce': 12,
#  'provide': 13, 'species': 14, 'the': 15, 'they': 16, 'various': 17, 'as': 18, 'be': 19,
#  'besides': 20, 'can': 21, 'etc': 22, 'give': 23, 'in': 24, 'leaves': 25, 'making': 26,
#  'medicines': 27, 'numerous': 28, 'perfumes': 29, 'purposes': 30, 'shade': 31, 'such': 32,
#  'summer': 33, 'their': 34, 'us': 35, 'used': 36, 'atmosphere': 37, 'cool': 38, 'help': 39,
#  'moreover': 40, 'our': 41}


#Bag of Words :  [[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)], [(4, 1), 
# (7, 1), (8, 1), (9, 1), (10, 1), (11, 1), (12, 1), (13, 1), (14, 1), (15, 1), (16, 1),
#  (17, 1)], [(4, 1), (8, 1), (15, 1), (16, 1), (18, 1), (19, 1), (20, 1), (21, 1), (22, 1),
#  (23, 1), (24, 1), (25, 1), (26, 1), (27, 1), (28, 1), (29, 1), (30, 1), (31, 1), (32, 1),
#  (33, 1), (34, 1), (35, 1), (36, 1)], [(16, 1), (37, 1), (38, 1), (39, 1), (40, 1),
#  (41, 1)], []]
