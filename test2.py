import pickle

word2id = pickle.load(open("dict.pkl", "rb"))
id2word = {v:k for k, v in word2id.items()}


poem = list()
with open("final.txt", "r") as f:
    for l in f.readlines():
        poem.append(l.split())

poems = []
for p in poem:
    pp = list(map(int, p))
    poems.append(pp)

for p in poems[:20]:
    i = 1
    line = ""
    for idx in p:
        word = id2word[idx]
        line += word
        if i % 5 == 0 :
            line += " "
        i += 1
    print(line)
    print()