from numpy import *
import re
import feedparser


# 测试数据集
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


# 创建词汇表
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


# 构建词汇向量
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word:%s is not in my Vocabulary!" % word)
    return returnVec


# 计算分类所需概率
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)
    return p0Vect, p1Vect, pAbusive


# 根据概率算法进行分类
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify*p1Vec)+log(pClass1)
    p0 = sum(vec2Classify*p0Vec)+log(1.0-pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


# 测试
def testingNB():
    listOposts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOposts)
    trainMat = []
    for postinDoc in listOposts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0v, p1v, pAb = trainNB0(trainMat, listClasses)
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as:', classifyNB(thisDoc, p0v, p1v, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as:', classifyNB(thisDoc, p0v, p1v, pAb))


def bagOfWordsVecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = +1
        else:
            print('the word:%s is not in my Vocabulary!' % word)
    return returnVec


# 文本处理
def textParse(bigString):
    regEx = re.compile(r'\W+')
    # file1 = open(bigString, 'rb').read()
    listOfTokens = regEx.split(bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = textParse('email/spam/%d.txt' % i)
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse('email/ham/%d.txt' % i)
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append((classList[docIndex]))
    p0V, p1V, pSpm = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordvector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordvector), p0V, p1V, pSpm) != classList[docIndex]:
            errorCount += 1
    print('the error rate is %.5f' % (float(errorCount)/len(testSet)))


def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]


def localWrods(feed1, feed0):
    docList = []
    classList = []
    fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList, fullText)
    for pariW in top30Words:
        if pariW[0] in vocabList:
            vocabList.remove(pariW[0])
    trainingSet = list(range(2*minLen))
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = []
    trainClass = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClass.append((classList[docIndex]))
    p0V, p1V, pSpm = trainNB0(array(trainMat), array(trainClass))
    errorCount = 0
    for docIndex in testSet:
        wordvector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordvector), p0V, p1V, pSpm) != classList[docIndex]:
            errorCount += 1
    print('the error rate is %.5f' % (float(errorCount) / len(testSet)))
    return vocabList, p0V, p1V


def getTopWords(ny,sf):
    vocabList, p0V, p1V = localWrods(ny, sf)
    topNY = []
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0:
            topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0:
            topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF")
    for item in sortedSF:
        print(item[0])
    sortedNF = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NF**NF**NF**NF**NF**NF**NF**NF")
    for item in sortedNF:
        print(item[0])


ny = feedparser.parse('http://www.nasa.gov/rss/dyn/image_of_the_day.rss')
sf = feedparser.parse('http://sports.yahoo.com/nba/teams/hou/rss.xml')
getTopWords(ny, sf)

# NASA Image of the Day：http://www.nasa.gov/rss/dyn/image_of_the_day.rss
# Yahoo Sports - NBA - Houston Rockets News：http://sports.yahoo.com/nba/teams/hou/rss.xml


# regEx = re.compile('\W+')
# emailText = open('email/ham/6.txt', 'rb').read()
# # print(emailText)
# listOfTokens = regEx.split(emailText.decode('utf-8'))
# print(listOfTokens)
# # testingNB()


# #
# listOposts, listClasses = loadDataSet()
# myVocabList = createVocabList(listOposts)
# print(myVocabList)
# trainMat = []
# for postinDoc in listOposts:
#     trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
# print(trainMat)
# p0v, p1v, pAb = trainNB0(trainMat, listClasses)
# print(pAb)




