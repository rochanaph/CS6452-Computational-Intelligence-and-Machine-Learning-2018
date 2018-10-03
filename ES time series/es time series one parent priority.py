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
        self.n = 4 # jumlah beta
        self.alfa = random.random() # sementara random biasa
        self.beta = np.random.normal(0, 1, self.n) # random gaussian list numpy
        self.tau  = random.gauss(0, 1)# random gaussian 1 nilai
        self.krom = np.hstack([self.alfa, self.beta])

def mutasi(individu, lamda=7):
    """
    fungsi untuk melakukan mutasi dengan rumus tau tunggal tanpa korelasi
    :param individu: object kromosom
    :param m: banyak child dari iterasi mutasi, defaultnya 7
    :return: list individu hasil mutasi
    """
    n = individu.n + 1
    #pm = np.random.normal(0, 1, n) # sementara semua gen kena mutasi
    tauBig = 1/math.sqrt(n)
    tau = individu.tau * math.exp(tauBig * random.gauss(0, 1))
    result = []
    for i in range(lamda):
        child = copy.deepcopy(individu)
        child.tau = tau
        child.krom = child.krom + (child.tau * random.gauss(0, 1))
        result.append(child)
    return result
def predict(model, data):
    """
    fungsi untuk melakukan prediksi dengan sliding windows
    :param model: susunan parameter regresi
    :param data: data nilai tukar rupiah yang dijadikan variabel independen
    :return: hasil prediksi
    """
    prediction = np.convolve(model[1:][::-1], data[:-1], "valid") + model[0]
    return prediction

def fitness(individu, data):
    """
    fungsi untuk menghitung nilai fitness dengan rumus 1/mse.
    :param individu: satu buah object kromosom
    :param data: data nilai tukar rupiah sebagai training parameter regresi dan perhitungan error
    :return: nilai fitness untuk satu individu
    """
    prediction = predict(individu.krom, data)
    real = data[individu.n:]
    mse = np.square(np.subtract(prediction, real)).mean()
    return 1/mse

def main():
    # read data, split train-test
    data = pd.read_csv("USD IDR Historical Data.csv", sep="\t")
    data["Date"] = pd.to_datetime(data['Date'])
    data.index = data["Date"]
    del (data["Date"])
    train = data['2017/03/6':]["Price"].str.replace(",", "").values
    train = train.astype(np.float32)
    test = data[:'2017/03/6']["Price"].str.replace(",", "").values
    test = test.astype(np.float32)

    # init generasi awal
    generasi = 100
    popSize  = 50
    selectRandom = True

    if selectRandom:
        print ("model forecasting dengan seleksi survivor random dari kandidat mutasi")
    else:
        print ("model forecasting dengan seleksi survivor best fitness dari kandidat mutasi")
    print ("popsize: ", popSize)


    # iterasi
    for g in range(generasi):
        # seleksi
        if g == 0:
            bestFitness = 0
            bestParent = []
            populasi = [kromosom() for i in range(popSize)]
        else:
            if selectRandom:
                randomIndex = random.sample(range(len(mutatedList)), popSize-1 )
                randomMutated = [mutatedList[i] for i in randomIndex]
                populasi = [bestParent] + randomMutated
            else:
                fitnessMutated = [fitness(mutated, train) for mutated in mutatedList]
                sortedIndex = np.argsort(fitnessMutated)[::-1][:popSize-1]
                sortedMutated = [mutatedList[i] for i in sortedIndex]
                populasi = [bestParent] + sortedMutated

        # mutasi
        mutatedList = []
        for p in range(popSize):
            mutatedList.extend(mutasi(populasi[p]))

        # fitness, menghitung fitness dari populasi
        fitnessList = []
        for parent in populasi:
            fitnessList.append(fitness(parent, train))

        # elitism
        bestIndex = np.argmax(fitnessList)
        if bestFitness < fitnessList[bestIndex] :
            bestFitness = fitnessList[bestIndex]
            bestParent = populasi[bestIndex]
        if g%10 == 0:
            print ("generasi ",g, " best fitness", bestFitness)

    print ("model regresi terbaik, dengan susunan alfa dan 4 beta:")
    print (bestParent.krom)
    # predict data test
    model = bestParent.krom
    prediction = predict(model, test)
    real = test[bestParent.n:]
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
    return bestParent

if __name__ == "__main__":
    main()