import numpy as np
import random
import math
import copy
import pandas as pd
import matplotlib.pyplot as plt

class kromosom:
    def __init__(self):
        """
        melakukan init untuk object kromosom
        alfa = parameter alfa pada regresi
        beta = parameter beta pada regresi, merupakan pengali data
        tau  = parameter mutasi untuk evolutionary strategy tunggal tanpa korelasi
        """
        self.n1 = 4 # jumlah param regresi untuk n hari
        self.n2 = 4
        self.alfa = random.random() # sementara random biasa
        self.beta = np.random.normal(0, 1, self.n1) # random gaussian list numpy
        self.gamma = np.random.normal(0, 1, self.n2)
        self.tau  = random.gauss(0, 1)# random gaussian 1 nilai
        self.krom = np.hstack([self.alfa, self.beta, self.gamma])

def mutasi(individu, lamda=7):
    """
    fungsi untuk melakukan mutasi dengan rumus tau tunggal tanpa korelasi
    :param individu: object kromosom
    :param m: banyak child dari iterasi mutasi, defaultnya 7
    :return: list individu hasil mutasi
    """
    n = individu.n1 + individu.n2 + 1
    tauBig = 1/math.sqrt(n)
    tau = individu.tau * math.exp(tauBig * random.gauss(0, 1))
    result = []
    for i in range(lamda):
        child = copy.deepcopy(individu)
        child.tau = tau
        child.krom = child.krom + (child.tau * random.gauss(0, 1))
        result.append(child)
    return result

def predict(model, data, data2):
    """
    fungsi untuk melakukan prediksi dengan sliding windows
    :param model: susunan parameter regresi
    :param data: data nilai tukar rupiah yang dijadikan variabel independen
    :param data2: data inflasi sebagai variabel tambahan
    :return: hasil prediksi
    """
    n = len(model)
    nBeta = int((n-1)/2)
    prediction = model[0] + np.convolve(model[1:nBeta+1][::-1], data[:-1], "valid") + np.convolve(model[nBeta+1:][::-1], data[:-1], "valid")
    return prediction

def fitness(individu, data, data2):
    """
    fungsi untuk menghitung nilai fitness dengan rumus 1/mse.
    :param individu: satu buah object kromosom
    :param data: data nilai tukar rupiah sebagai training parameter regresi dan perhitungan error
    :return: nilai fitness untuk satu individu
    """
    prediction = predict(individu.krom, data, data2)
    real = data[individu.n1:]
    mse = np.square(np.subtract(prediction, real)).mean()
    return (1/mse)

def main():
    # read data
    data = pd.read_csv("USD IDR Historical Data.csv", sep="\t")
    dataInflasi = pd.read_csv("Rate inflasi.csv", sep=",")

    # proses matching rate inflasi
    yearInflasi = dataInflasi["Year"].tolist()
    monthInflasi = dataInflasi["Month"].tolist()
    rateInflasi    = dataInflasi["Rate"].tolist()
    dictInflasi = {}
    for i in range(len(dataInflasi)):
        dictInflasi[str(yearInflasi[i])+"-"+monthInflasi[i]] = rateInflasi[i]

    tanggalExchange = data["Date"].tolist()
    inflasi = []
    for item in tanggalExchange:
        inflasi.append(dictInflasi[item.split("-")[2]+"-"+item.split("-")[1]])

    # split train test
    data["Date"] = pd.to_datetime(data['Date'])
    data.index = data["Date"]
    del (data["Date"])
    train = data['2017/03/6':]["Price"].str.replace(",","").values
    train = train.astype(np.float32)
    trainInflasi = np.array(inflasi[:len(train)]).astype(np.float32)
    test = data[:'2017/03/6']["Price"].str.replace(",","").values
    test = test.astype(np.float32)
    testInflasi = np.array(inflasi[len(train)-1:]).astype(np.float32)


    # init generasi awal
    generasi = 100
    popSize  = 50
    siklusPlus = False

    if siklusPlus:
        print ("model forecasting dgn tambahan data inflasi menggunakan siklus parent + hasil mutasi")
    else:
        print ("model forecasting dgn tambahan data inflasi menggunakan siklus dari hasil mutasi saja")
    print ("popsize: ", popSize)


    # iterasi
    for g in range(generasi):
        if g == 0:
            bestFitness = 0
            bestIndividu = []
            populasi = [kromosom() for i in range(popSize)]

        # mutasi
        mutatedList = []
        for p in range(popSize):
            mutatedList.extend(mutasi(populasi[p]))

        # fitness, menghitung fitness dari child hasil mutasi
        fitnessList = []
        for m in mutatedList:
            fitnessList.append(fitness(m, train, trainInflasi))

        # seleksi
        if siklusPlus:
            # candidate pop dari parent ditambah mutatedList (myu+lamda)
            candidateJoin = populasi + mutatedList
            fitnessParent = [fitness(parent, train, trainInflasi) for parent in populasi]
            fitnessJoin   = fitnessParent + fitnessList
            sortedIndex   = np.argsort(fitnessJoin)[::-1][:popSize]
            populasi      = [candidateJoin[i] for i in sortedIndex]

        else:
            # candidate pop dari mutatedList saja  (myu, lamda)
            fitnessJoin = fitnessList
            sortedIndex = np.argsort(fitnessJoin)[::-1][:popSize]
            populasi    = [mutatedList[i] for i in sortedIndex]

        # elitism, mencari dan menjaga individu terbaik
        bestIndex = sortedIndex[0]
        if bestFitness < fitnessJoin[bestIndex] :
            bestFitness = fitnessJoin[bestIndex]
            bestIndividu = populasi[0]
        if g%10 == 0:
            print ("generasi ",g, " best fitness", bestFitness)

    print ("model regresi terbaik, dengan susunan alfa, 4 beta data nilai tukar, 4 gamma data rate inflasi:")
    print (bestIndividu.krom)

    # predict data test
    model = bestIndividu.krom
    prediction = predict(model, test, testInflasi)
    real = test[bestIndividu.n1:]
    print ("contoh hasil prediksi 4 hari pertama data test")
    print (prediction[:4])
    print ("nilai tukar rupiah real 4 hari pertama data test")
    print (real[:4])

    # plot test prediction
    tanggal = data[:'2017/03/10'].index
    plt.plot(tanggal, prediction,
             tanggal, real)
    plt.gca().legend(('Prediction', 'Real'))
    plt.grid()
    plt.show()

    return 0

if __name__ == "__main__":
    main()