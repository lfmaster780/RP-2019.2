arquivo = open("base.csv","r", encoding = "utf-8")
arq = arquivo.readlines()
escrever = open("bag.dat","w")
conj = set()
for linha in arq:
    t = linha.split(";")
    citacao = t[0]
    citacao = citacao.split(" ")
    for palavra in citacao:
        conj.add(palavra.lower())


for k in conj:
    escrever.write(k+"\n")

escrever.close()
arquivo.close()
