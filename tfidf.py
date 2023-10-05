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


#topic 2:: TFIDF

import gensim
from gensim.utils import np, simple_preprocess
from gensim import corpora, models

#take th input
inp = ["The food is excellent but the service can be better",
        
        "The food was mediocre and the service was terrible"]

g_dict = corpora.Dictionary([simple_preprocess(line) for line in inp])
g_bow = [g_dict.doc2bow(simple_preprocess(line)) for line in inp]

print("Dictionary : ")
for item in g_bow:
    print([[g_dict[id], freq] for id, freq in item])

g_tfidf = models.TfidfModel(g_bow, smartirs='ntc')

print("TF-IDF Vector:")
for item in g_tfidf[g_bow]:
    print([[g_dict[id], np.around(freq, decimals=2)] for id, freq in item]

#output::

#Dictionary : 
#[['be', 1], ['better', 1], ['but', 1], ['can', 1], ['excellent', 1], ['food', 1], ['is', 1], ['service', 1], ['the', 2]]
#[['food', 1], ['is', 1], ['service', 1], ['the', 2], ['always', 1], ['and', 1], ['delicious', 1], ['loved', 1]]
#[['food', 1], ['service', 1], ['the', 2], ['and', 1], ['mediocre', 1], ['terrible', 1], ['was', 2]]
#TF-IDF Vector:
#[['be', 0.43], ['better', 0.43], ['but', 0.43], ['can', 0.43], ['excellent', 0.43], ['food', 0.09], ['is', 0.21], ['service', 0.09], ['the', 0.18]]
#[['food', 0.11], ['is', 0.26], ['service', 0.11], ['the', 0.21], ['always', 0.52], ['and', 0.26], ['delicious', 0.52], ['loved', 0.52]]
#[['food', 0.08], ['service', 0.08], ['the', 0.16], ['and', 0.2], ['mediocre', 0.39], ['terrible', 0.39], ['was', 0.78]]

#topic 3 :: word2vec

from gensim.models.word2vec import Word2vec
from multiprocessing import cpu_count

data = [
    "The food is excellent but the service can be better",
    "The food is always delicious and loved the service" 
]

tokenized_data = [sentence.split() for sentence in data]

w2v_model = word2vec[tokenized_data, min_count=0, workers=cpu_count()]

similar_words = w2v_model.wv.most_similar('word')
for word, score in similar_words:
    print(f"{word}: {score}")