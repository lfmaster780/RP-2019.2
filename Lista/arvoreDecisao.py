import matplotlib.pyplot as plt
import random

arquivo = open("base.csv","r", encoding = "utf8")
arq = arquivo.readlines()
tp = 0.4
zn = 1.96

classificacao = []
itens = []

for linha in arq:
    t = linha.split(";")
    citacao = t[0]
    clas = t[1].rstrip('\n')

    itens.append(citacao)
    classificacao.append(clas)


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

X = vectorizer.fit_transform(itens)

pontos = X.toarray()
bagOfWords = vectorizer.vocabulary_

from sklearn import preprocessing
lenc = preprocessing.LabelEncoder()
classificacoes = lenc.fit_transform(classificacao)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(pontos, classificacoes, test_size=tp, random_state=100)

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)

classes = list(lenc.inverse_transform(clf.classes_))
print("Taxa de acerto com",str(tp*100)+"%","base sendo teste:", str(clf.score(X_test, y_test) * 100) + "%\n")
numAmostras = len(pontos)

erroAmostral= 0

for i in range(numAmostras):
    if clf.predict(pontos[i].reshape(1, -1))[0] != classificacoes[i]:
        erroAmostral += 1 / numAmostras



sigma = ( zn * ( (erroAmostral * ( 1 - erroAmostral ) ) / numAmostras ) ** (1/2) )
print("Erro real:")
print("+", (erroAmostral + sigma) * 100, "%")
print("-", (erroAmostral - sigma) * 100, "%")

for i in range(len(X_test)):
    print()
    print("Exemplo ", i)
    classificado = lenc.inverse_transform(clf.predict(X_test[i].reshape(1, -1)))
    real = lenc.inverse_transform([y_test[i]])
    print("Classes:",classes)
    print("classificado como: ", classificado[0])
    print("real : ", real[0])
    if classificado[0] == real[0]:
        print("Acertou")
    else:
        print("Errou")

print("")
entrada = str(input())
while entrada != "0" and entrada != "":
    ent = []
    ent.append(entrada)
    print("Classificacao: ", end = "")

    ent = vectorizer.transform(ent).toarray()

    print(clf.predict(ent), end="")
    print(list(lenc.inverse_transform(clf.predict(ent))))
    entrada = str(input())

print("")
# from sklearn.externals.six import StringIO
# from IPython.display import Image
# from sklearn.tree import export_graphviz
# import pydotplus
# print("Gerando arvore")
# dot_data = StringIO()
# export_graphviz(clf, out_file=dot_data,
#                 filled=True, rounded=True,
#                 special_characters=True, feature_names = vectorizer.get_feature_names(),class_names=['0','1','2'])
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_png('arvore.png')
# Image(graph.create_png())
# arquivo.close()
