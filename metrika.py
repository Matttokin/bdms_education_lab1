import pandas as pd
from numpy import math


#
#блок функций
#

#расчет tf для строки | возвращает лист элементов и их tf-значения
def calcTf(text, meta):
  tfDictionary = {}
  text = text.split()
  for elem in meta:
    tf = text.count(elem)
    tfDictionary[elem] = tf/len(text)
  return tfDictionary

#расчет idf конкретного слова в тексте | возвращает idf-значение слова
def calcIdf(text, word):
    sumValue = sum([1.0 for i in text if word in i])
    if sumValue != 0.0:
        idfValue =  math.log10(len(text) / sumValue)
    else:
        idfValue = 0.0
    return  idfValue
def calcIdfFull(text, meta):
    idfDictionary = {}
    text = text.str.split().tolist()
    for elem in meta:
        idfDictionary[elem] = calcIdf(text, elem)
    return  idfDictionary

#расчет tf-idf для строки | возвращает лист элементов и их tf-idf-значения
def calcTfIdf(text, meta,idfD):
    tfDictionary = calcTf(text, meta)
    tfIdfDictionary = {}
    for elem in tfDictionary:
            tfIdfDictionary[elem] = tfDictionary[elem] * idfD[elem]
    return tfIdfDictionary

#расчет tf для строки | возвращает общее tf-значение
def calcTfFull(text, meta):
    tfDictionary = calcTf(text, meta)
    tfValue = 0.0
    for elem in tfDictionary:
        tfValue += tfDictionary[elem]
    return  tfValue

#расчет tf-idf для строки | возвращает общее tf-idf-значение
def calcTfIdfFull(text, meta,idfD):
    tfIdfDictionary = calcTfIdf(text, meta,idfD)
    tfIdfValue = 0.0
    for elem in tfIdfDictionary:
        tfIdfValue += tfIdfDictionary[elem]
    return  tfIdfValue

#расчет средней длины документа | возвращает общее значение
def calcAvgDoc(dataFrame):
    lengthAll = 0.0
    for doc in dataFrame["text"]:
        lengthAll += len(doc.split())
    return  lengthAll/len(dataFrame["text"])

#расчет bm25 для строки | возвращает элементов и их bm25-значения
def calcBm25(text, meta ,avgDoc,idfD):
  bm25Dictionary = {}
  texSplit = text.split()
  k = 2.0 # k и b свободные коэффициенты
  b = 0.75
  lenDoc = len(texSplit)
  for elem in meta:
    tf = texSplit.count(elem)
    bm25Dictionary[elem] = (idfD[elem] * tf *(k + 1)) /(tf + k * (1 - b + b * (lenDoc / avgDoc)))
  return bm25Dictionary

#расчет bm25 для строки | возвращает общее  bm25-значение
def calcBm25Full(text, meta ,avgDoc,idfD):
  bm25Dictionary = calcBm25(text,meta,avgDoc,idfD)
  bm25Value = 0.0
  for elem in bm25Dictionary:
      bm25Value += bm25Dictionary[elem]
  return bm25Value
#приведение dataFrame к нижнему регистру
def dataFrame2Lower(dataFrame):
    dataFrame["text"] = dataFrame["text"].apply(lambda row: row.lower())
    return dataFrame
#приведение list к нижнему регистру
def list2lower(list):
    return [x.lower() for x in list]

#функция для поиска документов по запросу searchType = maxValidity | minValidity | moreThanAverage .... countRow количество строк на вывод, если 0 или <0, то выводятся все строки
def searchFunc(dataFrame,searchRequest,searchType,countRow):
  dataFrame = dataFrame2Lower(dataFrame)
  searchRequest = list2lower(searchRequest)
  idfD = calcIdfFull(dataFrame["text"],searchRequest)
  avgDoc = calcAvgDoc(dataFrame)

  columnsTf = dataFrame["text"].apply(lambda row: pd.Series(calcTfFull(row, searchRequest), index=["tf"]))
  columnsTfIdf = dataFrame["text"].apply(lambda row: pd.Series(calcTfIdfFull(row,searchRequest,idfD),index=["tfidf"]))
  columnsBm25 = dataFrame["text"].apply(lambda row: pd.Series(calcBm25Full(row,searchRequest,avgDoc,idfD), index=["bm25"])) # проход по строкам и расчет

  dataFrameMerge = dataFrame
  dataFrameMerge["tf"] = columnsTf
  dataFrameMerge["tfidf"] = columnsTfIdf
  dataFrameMerge["bm25"] = columnsBm25

  if searchType == "maxValidity":
    print("Наиболее актуальные документы по запросу")
    if countRow > 0:
      print("searchByTf")
      print(dataFrameMerge.sort_values(by=['tf'],ascending=[False])["tf"][:countRow])
      print("-------------------------------")
      print("searchByTfIdf")
      print(dataFrameMerge.sort_values(by=['tfidf'],ascending=[False])["tfidf"][:countRow])
      print("-------------------------------")
      print("searchByBm25")
      print(dataFrameMerge.sort_values(by=['bm25'],ascending=[False])["bm25"][:countRow])
      print("-------------------------------")
    else:
      print("searchByTf")
      print(dataFrameMerge.sort_values(by=['tf'],ascending=[False])["tf"])
      print("-------------------------------")
      print("searchByTfIdf")
      print(dataFrameMerge.sort_values(by=['tfidf'],ascending=[False])["tfidf"])
      print("-------------------------------")
      print("searchByBm25")
      print(dataFrameMerge.sort_values(by=['bm25'],ascending=[False])["bm25"])
      print("-------------------------------")
  else:
    if searchType == "minValidity":
      print("Наименее актуальные документы по запросу")
      if countRow > 0:
        print("searchByTf")
        print(dataFrameMerge.sort_values(by=['tf'],ascending=[True])["tf"][:countRow])
        print("-------------------------------")
        print("searchByTfIdf")
        print(dataFrameMerge.sort_values(by=['tfidf'],ascending=[True])["tfidf"][:countRow])
        print("-------------------------------")
        print("searchByBm25")
        print(dataFrameMerge.sort_values(by=['bm25'],ascending=[True])["bm25"][:countRow])
        print("-------------------------------")
      else:
        print("searchByTf")
        print(dataFrameMerge.sort_values(by=['tf'],ascending=[True])["tf"])
        print("-------------------------------")
        print("searchByTfIdf")
        print(dataFrameMerge.sort_values(by=['tfidf'],ascending=[True])["tfidf"])
        print("-------------------------------")
        print("searchByBm25")
        print(dataFrameMerge.sort_values(by=['bm25'],ascending=[True])["bm25"])
        print("-------------------------------")
    else:
      if searchType == "moreThanAverage":
        print("Средние показатели")
        tfMean = dataFrameMerge["tf"].mean()
        tfIdfMean = dataFrameMerge["tfidf"].mean()
        bm25Mean = dataFrameMerge["bm25"].mean()
        print("tf",tfMean)
        print("tfIdf",tfIdfMean)
        print("bm25",bm25Mean)
        print("Документы по запросу  с показателем выше среднего")
        if countRow > 0:
          print("searchByTf")
          print(dataFrameMerge.sort_values(by=['tf'],ascending=[False])[dataFrameMerge.tf > tfMean]["tf"][:countRow])
          print("-------------------------------")
          print("searchByTfIdf")
          print(dataFrameMerge.sort_values(by=['tfidf'],ascending=[False])[dataFrameMerge.tfidf > tfIdfMean]["tfidf"][:countRow])
          print("-------------------------------")
          print("searchByBm25")
          print(dataFrameMerge.sort_values(by=['bm25'],ascending=[False])[dataFrameMerge.bm25 > bm25Mean]["bm25"][:countRow])
          print("-------------------------------")
        else:
          print("searchByTf")
          print(dataFrameMerge.sort_values(by=['tf'],ascending=[False])[dataFrameMerge.tf > tfMean]["tf"])
          print("-------------------------------")
          print("searchByTfIdf")
          print(dataFrameMerge.sort_values(by=['tfidf'],ascending=[False])[dataFrameMerge.tfidf > tfIdfMean]["tfidf"])
          print("-------------------------------")
          print("searchByBm25")
          print(dataFrameMerge.sort_values(by=['bm25'],ascending=[False])[dataFrameMerge.bm25 > bm25Mean]["bm25"])
          print("-------------------------------")





dataFrameNew = pd.DataFrame({
    "text":pd.Series(
        ["Покупка Samsung Galaxy S8",
         "Samsung Galaxy S8 по низкой цене",
         "Samsung Galaxy S8 объявления о продаже в Московской области",
         "Samsung Galaxy S8 Plus Gold - купить смартфон Самсунг‎",
         "Купить Смартфон Samsung Galaxy S8 64Gb Черный",
         "Samsung Galaxy S8 2 SIM – купить мобильный телефон",
         "Смартфон Samsung Galaxy S8 — купить по выгодной цене",
         "Купить Samsung Galaxy S8 black diamond в Москве, цена",
         "Купить смартфон Samsung Galaxy S8 и S8 Plus в Москве",
         "Новый Samsung Galaxy S8 | S8+ уже сейчас | Samsung ru",
         "Купить 5.8 Смартфон Samsung Galaxy S8 64 ГБ черный",
         "Смартфон Samsung Galaxy S8: купить по цене от 16449",
         "Купить оригинальный Samsung Galaxy S8 недорого"
         ])})


seachReq = "Купить Samsung Galaxy S8"

searchFunc(dataFrameNew,seachReq.split(),"maxValidity",5)

print(dataFrameNew.loc[0])
print(dataFrameNew.loc[12])

