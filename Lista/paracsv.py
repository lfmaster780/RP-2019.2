arquivo = open("friendship","r",encoding="utf8")
escrever = open("friendship.csv", "w",encoding="utf8")
classi = "friendship"
arq = arquivo.readlines()

for k in arq:
    g = k
    if (g == "\n" or g == "" or g == " "):
        continue
    g = g[0:len(g)-1]
    escrever.write(g+";"+classi+"\n")

escrever.close()
