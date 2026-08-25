import gensim.downloader as api

model = api.load("glove-wiki-gigaword-50")


print(model.most_similar([model['king']], topn=11))
print(model.most_similar([model['outro']], topn=11))
