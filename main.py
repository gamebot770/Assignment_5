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

def make_np_array_with_one_set_val(index, value, length_of_array):
    arr = np.zeros(length_of_array)
    arr[index] = value
    return arr

corpus = sc.textFile("s3://chrisjermainebucket/comp330_A5/SmallTrainingDataOneLinePerDoc.txt")

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
normalized_TF_IDF = normalized_TF_IDF.zipWithIndex()
n = normalized_TF_IDF.count()

#TODO: REVIEW THIS
# r = np.ones(20000) * 0.01
# l = 0.1
x = normalized_TF_IDF
x.cache()
labels = x.map(lambda x: (x[1], isCourtDoc(x[0][0]))).map(lambda x: make_np_array_with_one_set_val(x[0], x[1], 20000)).reduce(lambda acc, ele: np.add(acc, ele))
y = labels
print(x.filter(lambda x: isCourtDoc(x[0][0]) != y[x[1]]).collect())

# theta = x.map(lambda x: (x[0][0], np.dot(x[0][1], r))).fold(('', np.array([])), lambda acc, ele: (acc[0], np.append(acc[1], ele[1])))[1]
# e_to_the_theta = np.exp(theta)
# one_plus_e_to_the_theta = 1 + e_to_the_theta
# log_one_plus_e_to_the_theta = np.log(1 + e_to_the_theta)
# error = np.linalg.norm(r, ord=2) * l
# negative_y_theta = y * -1 * theta
# total_vector = negative_y_theta + log_one_plus_e_to_the_theta + error
# total = np.sum(total_vector)
#
# #gradient
# negative_y_x = x.map(lambda x: ((x[0][0], x[0][1] * y[x[1]] * -1), x[1]))
# J = np.divide(e_to_the_theta, one_plus_e_to_the_theta)
#
# P1 = x.map(lambda x: ((x[0][0], x[0][1] * y[x[1]] * -1)))
# P2 = x.map(lambda x: ((x[0][0], x[0][1] * J[x[1]])))
# P3 = x.map(lambda x: ((x[0][0], l * r[x[1]])))
#
# gradient_raw = P1.join(P2).join(P3).map(lambda x: (x[0], x[1][0][0] + x[1][0][1] + x[1][1]))
# new_gradient = gradient_raw.fold(('', np.zeros(20000)), lambda acc, ele: (acc[0], acc[1] + ele[1]))[1]



def f(x, y, r, l):
    size_of_x = x.count()
    theta = x.map(lambda x: (x[1], np.dot(x[0][1], r))).map(lambda x_ele: make_np_array_with_one_set_val(x_ele[0], x_ele[1], size_of_x)).reduce(lambda acc, ele: np.add(acc, ele))
    e_to_the_theta = np.exp(theta)
    one_plus_e_to_the_theta = 1 + e_to_the_theta
    log_one_plus_e_to_the_theta = np.log(one_plus_e_to_the_theta)
    error = np.square(np.linalg.norm(r, ord=2)) * l
    negative_y_theta = y * -1 * theta
    total_vector = negative_y_theta + log_one_plus_e_to_the_theta + error
    total = np.sum(total_vector)
    return total


def gradient(x, y, r, l):
    size_of_x = x.count()
    theta = x.map(lambda x: (x[1], np.dot(x[0][1], r))).map(
        lambda x: make_np_array_with_one_set_val(x[0], x[1], size_of_x)).reduce(lambda acc, ele: np.add(acc, ele))
    e_to_the_theta = np.exp(theta)
    one_plus_e_to_the_theta = 1 + e_to_the_theta
    J = np.divide(e_to_the_theta, one_plus_e_to_the_theta)
    P1 = x.map(lambda x: ((x[0][0], x[0][1] * y[x[1]] * -1)))
    P2 = x.map(lambda x: ((x[0][0], x[0][1] * J[x[1]])))
    P3 = x.map(lambda x: ((x[0][0], 2 * l * r[x[1]])))
    gradient_raw = P1.join(P2).join(P3).map(lambda x: (x[0], x[1][0][0] + x[1][0][1] + x[1][1]))
    new_gradient = gradient_raw.fold(('', np.zeros(20000)), lambda acc, ele: (acc[0], acc[1] + ele[1]))[1]
    return new_gradient


def gd_optimize (x, y, w, c):
    rate = 1
    w_last = w + np.full (20000, 1.0)
    while (abs(f(x, y, w, c) - f(x, y, w_last, c)) > 10e-4):
        w_last = w
        print(gradient(x, y, w, c))
        w = w - rate * gradient (x, y, w, c)
        if f (x, y, w, c) > f (x, y, w_last, c):
            rate = rate * .5
        else:
            rate = rate * 1.1
        print (f (x, y, w, c))
        results = training_data.map(lambda x: (np.dot(x[0][1], w) > 0.001)).map(lambda x: (int(x))).fold(np.array([]), lambda acc, ele: np.append(acc, ele))
        print(results.size)
        total_correct_guesses = np.logical_and(results, y_training).sum()
        total_errors = np.logical_xor(results, y_training).sum()
        print("CORRECT: total_correct_guesses ", total_correct_guesses, " errors", total_errors)
    return w




# predictLabel(3, "Where do the gods rest upon the shoulders of man. How can we deny his existence?")

number_of_training = 2000
y_training = y[:number_of_training]
y_testing = y[number_of_training:]
modified_x_data =  x.map(lambda x: ((x[0][0], x[0][1] * 0.0001), x[1]))
training_data = modified_x_data.filter(lambda x: x[1] < number_of_training)
testing_data = modified_x_data.filter(lambda x: x[1] >= number_of_training)

training_data.cache()
testing_data.cache()
modified_x_data.cache()

optimized = gd_optimize(training_data, y_training, np.random.normal(0, 0.0001, 20000), 0.005)

results = testing_data.map(lambda x: (np.dot(x[0][1], optimized) > 0.001)).map(lambda x: (int(x))).fold(np.array([]), lambda acc, ele: np.append(acc, ele))

total_true = y.sum()
total_correct_guesses = np.logical_and(results, y_testing).sum()
total_errors = np.logical_xor(results, y_testing).sum()

print("CORRECT: total_correct_guesses ", total_correct_guesses)
print("WRONG: ", total_errors)

topFiftyIndices = np.argsort(optimized * -1)[:50]
topFiftyWordsForAUCases = dictionary.filter(lambda x: x[1] in topFiftyIndices).take(50)

#Top 50 Words
"""
[('that', 12), ('not', 28), ('mr', 206), ('applicant', 347), ('decision', 360), ('evidence', 365), ('whether', 430), ('tribunal', 479), ('respondent', 710), ('appeal', 751), ('reasons', 946), ('costs', 987), ('respect', 1077), ('ltd', 1141), ('relation', 1145), ('notice', 1148), ('sought', 1197), ('conduct', 1228), ('relevant', 1243), ('circumstances', 1320), ('honour', 1491), ('proceedings', 1772), ('error', 2227), ('satisfied', 2321), ('judgment', 2322), ('pty', 2327), ('pursuant', 2926), ('submissions', 2996), ('respondents', 3082), ('affidavit', 3122), ('counsel', 3299), ('fca', 3600), ('discretion', 3867), ('hca', 3887), ('clr', 3982), ('relied', 4048), ('fcr', 4273), ('multicultural', 4755), ('delegate', 4991), ('jj', 5172), ('jurisdictional', 6169), ('alr', 6322), ('fairness', 6611), ('interlocutory', 6729), ('relevantly', 7042), ('factual', 7618), ('fcafc', 8083), ('affidavits', 9238), ('gummow', 9669), ('gaudron', 17308)]
"""