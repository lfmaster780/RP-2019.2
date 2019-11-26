arquivo = open("base.csv","r", encoding = "utf8")
arq = arquivo.readlines()
tp = 0.4
zn = 1.96
splits = 3

classificacao = []
itens = []

for linha in arq:
    t = linha.split(";")
    citacao = t[0]
    clas = t[1].rstrip('\n')

    itens.append(citacao)
    classificacao.append(clas)

#Import Vectorizer para aplicar bag of words
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(itens)

pontos = X.toarray()
bagOfWords = vectorizer.vocabulary_

from sklearn import preprocessing

lenc = preprocessing.LabelEncoder()
classificacoes = lenc.fit_transform(classificacao)

from sklearn.naive_bayes import BernoulliNB
model = BernoulliNB()

#Dividir treino e teste
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(pontos, classificacoes, test_size=tp, random_state=100)
model.fit(X_train, y_train)

classes = list(lenc.inverse_transform(model.classes_))
#Prob a priori
priores = list(model.class_log_prior_)



for i in range(len(classes)):
    print("Classe:", classes[i], "Prob priori: ", priores[i])

print("Taxa de acerto com",str(tp*100)+"%","base sendo teste: ", model.score(X_test, y_test) * 100, "%")

print("Erro real:")
numAmostras = len(pontos)

erroAmostral= 0

for i in range(numAmostras):
    if model.predict(pontos[i].reshape(1, -1))[0] != classificacoes[i]:
        erroAmostral += 1 / numAmostras



sigma = ( zn * ( (erroAmostral * ( 1 - erroAmostral ) ) / numAmostras ) ** (1/2) )

print("+", (erroAmostral + sigma) * 100, "%")
print("-", (erroAmostral - sigma) * 100, "%")

for i in range(len(X_test)):
    print()
    print("Exemplo ", i)
    classificado = lenc.inverse_transform(model.predict(X_test[i].reshape(1, -1)))
    real = lenc.inverse_transform([y_test[i]])
    print("Classes:",classes)
    print("Probabilidades:", model.predict_proba(X_test[i].reshape(1, -1)))
    print("classificado como: ", classificado[0])
    print("real : ", real[0])
    if classificado[0] == real[0]:
        print("Acertou")
    else:
        print("Errou")

print("\nAplicando cross validation\n")
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, pontos, classificacoes, cv=splits)
print("Com",splits,"splits")
print("Taxa de acerto:", scores.mean()*100, "%, desvio de +/- ", scores.std() * 2 * 100, "%\n")

entrada = str(input())
while entrada != "0" and entrada != "":
    vecE = CountVectorizer(vocabulary=bagOfWords)
    entrada = vecE.fit_transform([entrada]).toarray()

    print (list(lenc.inverse_transform(model.classes_)))
    print ("Probabilidade:", model.predict_proba(entrada))
    print ("Predicted Class:", list(lenc.inverse_transform(model.predict(entrada))))
    entrada = str(input())

arquivo.close()
