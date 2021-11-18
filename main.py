import re
import numpy as np
import pyspark
import copy
# load up all of the 19997 documents in the corpus
sc = pyspark.SparkContext()

def countIndices(listOfPositions):
    accumulator = defaultdict(int)
    for i in listOfPositions:
        accumulator[i] += 1
    result = np.zeros(20000)
    for i in range(20000):
        result[i] = accumulator[i]
    return result


def divide(a, b):
    return a / b

def divideNumByArrayelement(a, b):
    return b / a

def incrementLabel(i, arr):
    arr[i] += 1

def keyPairsToNumpyArray(a):
    result = np.zeros(20000)
    for group in a:
        index = group[1][0]
        val = group[1][1]
        result[index] = val
    return result

def mse_alter(a, b):
    from numpy.linalg import norm
    diff = a - b
    sqr = np.square(diff)
    totals = sqr.sum()
    return np.sqrt(totals)

def mse(a, b):
    from numpy.linalg import norm
    diff = a - b
    return norm(diff)

def isCourtDoc(docID):
    if docID[:2] == "AU":
        return 1
    return 0


corpus = sc.textFile("SmallTrainingDataOneLinePerDoc.txt")

# each entry in validLines will be a line from the text file
validLines = corpus.filter(lambda x : 'id' in x)

print("SIZE {0}\n\n".format(validLines.__sizeof__()))
# now we transform it into a bunch of (docID, text) pairs
import re
import numpy as np
from collections import defaultdict

keyAndText = validLines.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:]))

# now we split the text in each (docID, text) pair into a list of words
# after this, we have a data set with (docID, ["word1", "word2", "word3", ...])
# we have a bit of fancy regular expression stuff here to make sure that we do not
# die on some of the documents
regex = re.compile('[^a-zA-Z]')
keyAndListOfWords = keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))

# now get the top 20,000 words... first change (docID, ["word1", "word2", "word3", ...])
# to ("word1", 1) ("word2", 1)...
allWords = keyAndListOfWords.flatMap(lambda x: ((j, 1) for j in x[1]))

# now, count all of the words, giving us ("word1", 1433), ("word2", 3423423), etc.
allCounts = allWords.reduceByKey (lambda a, b: a + b)

# and get the top 20,000 words in a local array
# each entry is a ("word1", count) pair
topWords = allCounts.top (20000, lambda x : x[1])

# and we'll create a RDD that has a bunch of (word, dictNum) pairs
# start by creating an RDD that has the number 0 thru 20000
# 20000 is the number of words that will be in our dictionary
twentyK = sc.parallelize(range(20000))

# now, we transform (0), (1), (2), ... to ("mostcommonword", 1) ("nextmostcommon", 2), ...
# the number will be the spot in the dictionary used to tell us where the word is located
# HINT: make use of topWords in the lambda that you supply
dictionary = twentyK.map (lambda x: (topWords[x][0], x))

wordAndDocPairs = keyAndListOfWords.flatMap(lambda x: [(t, x[0]) for t in x[1]])
combinedWordDocInfo = wordAndDocPairs.join(dictionary)
listOfAllDictPos = combinedWordDocInfo.map(lambda x: (x[1][0], [x[1][1]])).reduceByKey(lambda a,b: a + b) # doc -> [indices]
listOfWordCountsByDocument = listOfAllDictPos.map(lambda x: (x[0], countIndices(x[1])))

devvec = np.vectorize(divide)
docToTotalWordCount = listOfWordCountsByDocument.map(lambda x: (x[0], x[1].sum()))


combinedWordcountInfo = listOfWordCountsByDocument.join(docToTotalWordCount)
TF = combinedWordcountInfo.map(lambda x: (x[0], devvec(x[1][0], x[1][1]))) #doc -> [answers]

wordDocumentCounts = listOfWordCountsByDocument.map(lambda x: (0, np.clip(x[1], 0, 1))).reduceByKey(lambda a, b: np.add(a, b)).map(lambda x: x[1])
divNumArray = np.vectorize(divideNumByArrayelement)
corpusSize = keyAndListOfWords.count()
IDF = wordDocumentCounts.map(lambda x: divNumArray(x, corpusSize)).map(lambda x: np.log(x)).take(1)[0]
TF_IDF = TF.map(lambda x: (x[0], np.multiply(x[1], IDF)))

means = TF_IDF.fold(('', np.zeros(20000)), lambda acc, ele: (acc[0], acc[1] + ele[1]))[1] / TF_IDF.count() #Will return the sum of all the rows in the matrix

sd_temp = TF_IDF.map(lambda x: (x[0], np.power((x[1] - means), 2))).fold(('', np.zeros(20000)), lambda acc, ele: (acc[0], acc[1] + ele[1]))[1] * (1/TF_IDF.count())
sd = np.power(sd_temp, 0.5)
sd[sd == 0] = 1

normalized_TF_IDF = TF_IDF.mapValues(lambda x: np.divide((x - means), sd))
n = normalized_TF_IDF.count()

#TODO: REVIEW THIS
r = np.ones(20000) * 0.01
l = 0.1
x = normalized_TF_IDF
labels = normalized_TF_IDF.map(lambda x: (x[0], isCourtDoc(x[0])))
y = labels

theta = None
e_to_the_theta = None
one_plus_e_to_the_theta = None
log_one_plus_e_to_the_theta = None
J = None
# error = np.linalg.norm(r, ord=2) * l
# negative_y_theta = y.join(theta).mapValues(lambda v: v[0] * v[1] * -1)
# total_vector = negative_y_theta.join(log_one_plus_e_to_the_theta).mapValues(lambda v: v[0] + v[1] + error)
# total = total_vector.fold(('', 0), lambda acc, ele: (acc[0], acc[1] + ele[1]))

#gradient
# negative_y_x = y.join(x).mapValues(lambda v: v[0] * v[1] * -1)
# J = np.divide(e_to_the_theta, one_plus_e_to_the_theta)
# P1 = y.join(x).mapValues(lambda v: v[0] * v[1] * -1)
# P2 = J.join(x).mapValues(lambda v: v[0] * v[1])
# P3 = 2 * l * r
# gradient_raw = P1.join(P2).mapValues(lambda v: v[0] + v[1] + P3)
# new_gradient = gradient_raw.fold(('', np.zeros(20000)), lambda acc, ele: (acc[0], acc[1] + ele[1]))[1]

def f(x, y, r, l):
    theta = x.map(lambda x: (x[0], np.dot(x[1], r)))
    e_to_the_theta = theta.map(lambda x: (x[0], np.exp(x[1])))
    one_plus_e_to_the_theta = e_to_the_theta.mapValues(lambda v: v + 1)
    log_one_plus_e_to_the_theta = one_plus_e_to_the_theta.mapValues(lambda v: np.log(v))
    error = np.linalg.norm(r, ord=2) * l
    negative_y_theta = y.join(theta).mapValues(lambda v: v[0] * v[1] * -1)
    total_vector = negative_y_theta.join(log_one_plus_e_to_the_theta).mapValues(lambda v: v[0] + v[1] + error)
    total = total_vector.fold(('', 0), lambda acc, ele: (acc[0], acc[1] + ele[1]))
    return total[1]

def gradient(x, y, r, l):
    theta = x.map(lambda x: (x[0], np.dot(x[1], r)))
    e_to_the_theta = theta.map(lambda x: (x[0], np.exp(x[1])))
    one_plus_e_to_the_theta = e_to_the_theta.mapValues(lambda v: v + 1)
    J = e_to_the_theta.join(one_plus_e_to_the_theta).mapValues(lambda v: v[0] / float(v[1]))
    P1 = y.join(x).mapValues(lambda v: v[0] * v[1] * -1)
    P2 = J.join(x).mapValues(lambda v: v[0] * v[1])
    P3 = 2 * l * r
    gradient_raw = P1.join(P2).mapValues(lambda v: v[0] + v[1] + P3)
    new_gradient = gradient_raw.fold(('', np.zeros(20000)), lambda acc, ele: (acc[0], acc[1] + ele[1]))[1]
    return new_gradient


def gd_optimize (x, y, w, c):
    rate = 1
    w_last = w + np.full (20000, 1.0)

    while (abs(f(x, y, w, c) - f(x, y, w_last, c)) > 10e-4):
         w_last = w
         # print(gradient(x, y, w, c))
         w = w - rate * gradient (x, y, w, c)
         if f (x, y, w, c) > f (x, y, w_last, c):
              rate = rate * .5
         else:
              rate = rate * 1.1
         print (f (x, y, w, c))
    return w



def predictLabel(k, inputString):
    words = sc.parallelize(inputString.split(" "))
    wordsPairs = words.map(lambda x: (x, 0))
    wordsToIndices= wordsPairs.join(dictionary).map(lambda x: (x[0], x[1][1]))
    groupedWordsForNumpyIndexing = wordsToIndices.map(lambda x: (x[1], 1)).reduceByKey(lambda a, b: a + b).map(lambda x: ("group", x)).groupBy(lambda x: x[0]).mapValues(list)
    wordCounts = groupedWordsForNumpyIndexing.map(lambda x: keyPairsToNumpyArray(x[1]))
    docSize = wordCounts.map(lambda x: x.sum()).take(1)[0]
    TF_INPUT = wordCounts.map(lambda x: devvec(x, docSize)).take(1)[0]
    TF_IDF_INPUT = TF_INPUT * IDF
    distances = TF_IDF.map(lambda x: (x[0], mse_alter(x[1], TF_IDF_INPUT))).cache()
    top_documents = distances.map(lambda x: (x[0].split("/")[1], 1)).takeOrdered(10, lambda x: x[1])
    documentSums = sc.parallelize(distances.map(lambda x: (x[0].split("/")[1], (x[1], 1))).takeOrdered(10, lambda x: x[1]))
    finalOptions = documentSums.reduceByKey(lambda a, b: (min(a[0], b[0]), a[1] + b[1])).take(10)
    finalResults = defaultdict(list)
    top_count = -float('inf')
    for item in finalOptions:
        count = item[1][1]
        doc = item[0]
        score = item[1][0]
        top_count = max(top_count, count)
        finalResults[count].append((score, doc))
    final_candidates = finalResults[top_count]
    print(min(final_candidates, key=lambda x: x[0])[1])
    print(finalOptions)



    # a = TF_IDF.lookup("20_newsgroups/comp.graphics/37261")[0]
    # b = TF_IDF.lookup("20_newsgroups/talk.politics.mideast/75944")[0]
    # c = TF_IDF.lookup("20_newsgroups/sci.med/58763")[0]

predictLabel(3, "Where do the gods rest upon the shoulders of man. How can we deny his existence?")

number_of_training = 1300
y_training = labels.sample(False, 0.4,0)
y_testing = labels.subtract(y_training)
modified_x_data =  x.map(lambda x: (x[0], x[1] * 0.001))
training_data = modified_x_data.join(y_training).mapValues(lambda v: v[0])
testing_data = modified_x_data.join(y_testing).mapValues(lambda v: v[0])

training_data.cache()
testing_data.cache()
modified_x_data.cache()

optimized = gd_optimize(training_data, y_training, r, 0.01)

results = testing_data.map(lambda x: (np.dot(x[0][1], optimized) > 0)).map(lambda x: (int(x))).fold(np.array([]), lambda acc, ele: np.append(acc, ele))

total_true = y.sum()
total_correct_guesses = np.logical_and(results, y_testing).sum()
total_errors = np.logical_xor(results, y_testing).sum()

print("CORRECT: total_correct_guesses ", total_correct_guesses)
print("WRONG: ", total_errors)