import matplotlib.pyplot as plt
import random

arquivo = open("base.csv","r", encoding = "utf8")
arq = arquivo.readlines()
tp = 0.25
zn = 1.96
splits = 5
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
#Utiliza metodo Gini (Metodo para otimizar os erros de classificacao)
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

print("\nAplicando cross validation\n")
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, pontos, classificacoes, cv=splits)
print("Com",splits,"splits")
print("Taxa de Acerto:", str(scores.mean()*100) + "%, desvio de +/- ", scores.std() * 2 * 100, "%\n")


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

print("Gerando arvore construida")
tree.plot_tree(clf, filled=True)
plt.show()
