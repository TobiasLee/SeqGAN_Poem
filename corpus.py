# import tensorflow as tf
import os
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
# turn chinese word into idx token an build a dict

import pickle



with open("poem.txt", "r") as f:
    print(f.readline())
# dict = pickle.load(open("dict.pkl", 'rb'))
# print(max(dict.values()))
# poem = [3959, 1437 ,6313 ,1475, 1218 ,410, 5503 ,1017, 3683 ,1308, 4358, 2747, 536, 1278 ,5610, 3943, 1243 ,3679 ,6317 ,6525]
# # poem = [3951, 5009, 1101, 1619, 2238, 1018, 984,574,5816,2423,6186,6337,2783,2163,6472,1808,3165 ,113 ,6519 ,1437]
# word2id = np.load("dict.pkl")
# id2word = {v:k for k, v in word2id.items()}
# for idx in poem:
#     print(id2word[idx])

# 5000 首唐诗
# corpus = "qts_tab.txt"
# poems = []
# with open(corpus, 'r', encoding='utf-8') as f:
#     # tags = f.readline().strip().split(u'\t')
#     for l in f.readlines():
#         poem = l.strip()
#         # print(len(poem))
#         if len(poem) == 9:
#             poems.append(poem)
#         # print(len(l))
#
# print(len(poems))
# #
# # print(poems)
#
# poems = poems[:290000]
# counter = CountVectorizer(token_pattern='(?u)\\b\\w+\\b')
# x = counter.fit_transform(poems)
# print()
# print()
# xx = np.argwhere(x)
# idx= xx[:, 1]
# idx = idx.reshape(-1, 5)
# print(idx.shape)
# idx = idx[:20000, :]
# # print(idx)
# flip = np.fliplr(idx)
# ts = flip.reshape(-1, 20)
# print(len(ts))
# np.savetxt("train.txt", ts, "%d")
# # np.save("dict.npy", counter.vocabulary_)
# print(type(counter.vocabulary_))
# f = open("dict.pkl","wb")
# pickle.dump(counter.vocabulary_,f)
# f.close()