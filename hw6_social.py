"""
Social Media Analytics Project
Name:
Roll Number:
"""

import hw6_social_tests as test

project = "Social" # don't edit this

### PART 1 ###

import pandas as pd
import nltk
from collections import Counter
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
endChars = [ " ", "\n", "#", ".", ",", "?", "!", ":", ";", ")" ]

'''
makeDataFrame(filename)
#3 [Check6-1]
Parameters: str
Returns: dataframe
'''
def makeDataFrame(filename):
    df = pd.read_csv(filename)
    return df


'''
parseName(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
def parseName(fromString):
    s1 = " "
    s2 = " "
    line = fromString.split( " (")
    s2 = line[0]
    s1 = s2.split(": ")
    return s1[1]


'''
parsePosition(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
def parsePosition(fromString):
    s1 = " "
    s2 = " "
    line = fromString.split(" (")
    s2 = line[1]
    s1 = s2.split(" ")
    return s1[0]


'''
parseState(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
def parseState(fromString):
    place = " "
    s1 = " "
    s2 = " "
    s3 = " "
    line = fromString.split(" (")
    s1 = line[1]
    s2 = s1.split("from ")
    s3 = s2[1]
    place = s3.split(")")
    return place[0]


'''
findHashtags(message)
#5 [Check6-1]
Parameters: str
Returns: list of strs
'''
def findHashtags(message):
    hash_lst = []
    no = message.count("#")
    string = message.split('#')
    for i in string[1:len(string)]:
        name = ''
        for j in i:
            if j not in endChars:
                name += j
            else:
                break
        name = '#' + name
        hash_lst.append(name)
    return hash_lst


'''
getRegionFromState(stateDf, state)
#6 [Check6-1]
Parameters: dataframe ; str
Returns: str
'''
def getRegionFromState(stateDf, state):
    row = stateDf.loc[stateDf['state'] == state, 'region']
    return row.values[0]


'''
addColumns(data, stateDf)
#7 [Check6-1]
Parameters: dataframe ; dataframe
Returns: None
'''
def addColumns(data, stateDf):
    names=[]
    positions=[]
    states=[]
    regions=[]
    hashtags=[]
    for index,row in data.iterrows():
        l = data["label"].loc[index]
        name=parseName(l)
        position=parsePosition(l)
        state=parseState(l)
        region=getRegionFromState(stateDf,state)
        t = data["text"].loc[index]
        hashtag=findHashtags(t)
        names.append(name)
        positions.append(position)
        states.append(state)
        regions.append(region)
        hashtags.append(hashtag)
    data["name"]=names
    data["position"]=positions
    data["state"]=states
    data["region"]=regions
    data["hashtags"]=hashtags
    return None




### PART 2 ###

'''
findSentiment(classifier, message)
#1 [Check6-2]
Parameters: SentimentIntensityAnalyzer ; str
Returns: str
'''
def findSentiment(classifier, message):
    score = classifier.polarity_scores(message)['compound']
    if score < -0.1:
        text = "negative"
    elif (score > 0.1):
        text = "positive"
    else:
        text = "neutral"
    return text


'''
addSentimentColumn(data)
#2 [Check6-2]
Parameters: dataframe
Returns: None
'''
def addSentimentColumn(data):
    classifier = SentimentIntensityAnalyzer()
    sentiments = []
    for index , row in data.iterrows():
        response = row["text"]
        sentiments.append(findSentiment(classifier,response))
    data["sentiment"] = sentiments
    return None


'''
getDataCountByState(data, colName, dataToCount)
#3 [Check6-2]
Parameters: dataframe ; str ; str
Returns: dict mapping strs to ints
'''
def getDataCountByState(data, colName, dataToCount):
    dictionary = {}
    for index, row in data.iterrows():
        if ((len(colName)==0 and len(dataToCount) == 0)) :
            state = row['state']
            if state  not in dictionary:
                dictionary[state] = 0
            dictionary[state] += 1
        elif(row[colName] == dataToCount):
            state = row['state']
            if state  not in dictionary:
                dictionary[state] = 0
            dictionary[state] += 1
    return dictionary


'''
getDataForRegion(data, colName)
#4 [Check6-2]
Parameters: dataframe ; str
Returns: dict mapping strs to (dicts mapping strs to ints)
'''
def getDataForRegion(data, colName):
    dictionary = {}
    for index, row in data.iterrows():
        if row['region'] not in dictionary:
            dictionary[row['region']] = {}
        if row[colName] not in dictionary[row['region']]:
            dictionary[row['region']][row[colName]] = 0
        dictionary[row['region']][row[colName]] += 1
    return dictionary




'''
getHashtagRates(data)
#5 [Check6-2]
Parameters: dataframe
Returns: dict mapping strs to ints
'''
def getHashtagRates(data):
    dictionary = {}
    for index,row in data.iterrows():
        lst = row["hashtags"]
        for tag in lst:
            if tag not in dictionary:
                dictionary[tag] = 0
            dictionary[tag] += 1

    return dictionary


'''
mostCommonHashtags(hashtags, count)
#6 [Check6-2]
Parameters: dict mapping strs to ints ; int
Returns: dict mapping strs to ints
'''
def mostCommonHashtags(hashtags, count):
    dictionary = {}
    while(len(dictionary)<count):
        maximum = 0
        for value in hashtags:
            if value not in dictionary:
                if(hashtags[value]>maximum):
                    maximum = hashtags[value]
                    high = value
        dictionary[high] = maximum
    return dictionary



'''
getHashtagSentiment(data, hashtag)
#7 [Check6-2]
Parameters: dataframe ; str
Returns: float
'''
def getHashtagSentiment(data, hashtag):
    sum_sentiments = 0
    total = 0
    for index, row in data.iterrows():
        msg = row['text']
        if hashtag in msg:
            total +=1
            if row['sentiment'] == 'positive':
                sum_sentiments += 1
            if row['sentiment'] == 'negative':
                sum_sentiments -= 1
    return (sum_sentiments/total)




### PART 3 ###

'''
graphStateCounts(stateCounts, title)
#2 [Hw6]
Parameters: dict mapping strs to ints ; str
Returns: None
'''
def graphStateCounts(stateCounts, title):
    import matplotlib.pyplot as plt
    lst = list(stateCounts.items())
    for key,value in lst:
        labels = key
        yValues = value
        plt.bar(labels,yValues,color='blue')
        plt.xlabel(title,loc='center')
        plt.xticks(rotation="vertical")
        plt.title(title)
    plt.show()
    return



'''
graphTopNStates(stateCounts, stateFeatureCounts, n, title)
#3 [Hw6]
Parameters: dict mapping strs to ints ; dict mapping strs to ints ; int ; str
Returns: None
'''
def graphTopNStates(stateCounts, stateFeatureCounts, n, title):
    featurerate = {}
    for i in stateFeatureCounts:
        featurerate[i] = (stateFeatureCounts[i] / stateCounts[i])
    graphStateCounts(dict(Counter(featurerate).most_common(5)), "Top n Feature")
    return


'''
graphRegionComparison(regionDicts, title)
#4 [Hw6]
Parameters: dict mapping strs to (dicts mapping strs to ints) ; str
Returns: None
'''
def graphRegionComparison(regionDicts, title):
    features = []
    regions = []
    region_feature = []
    for i in regionDicts:
        for j in regionDicts[i]:
            if i not in features:
                features.append(j)
    for i in regionDicts.keys():
        regions.append(i)
    for i in regionDicts:
        list = []
        for j in features:
            count = 0
            for k in regionDicts[i]:
                if j==k:
                    count += regionDicts[i][k]
            list.append(count)
        region_feature.append(list)
    sideBySideBarPlots(features, regions, region_feature, 'Region Comparison')
    return None



'''
graphHashtagSentimentByFrequency(data)
#4 [Hw6]
Parameters: dataframe
Returns: None
'''
def graphHashtagSentimentByFrequency(data):
    hash = getHashtagRates(data)
    hashtags = []
    frequencies = []
    sentiment_scores = []
    dictionary = mostCommonHashtags(hash, 50)
    for i in dictionary:
        sentiment_score = getHashtagSentiment(data, i)
        hashtags.append(i)
        frequencies.append(dictionary[i])
        sentiment_scores.append(sentiment_score)
    scatterPlot(frequencies, sentiment_scores, hashtags, 'Hashtags by Frequency & Sentiments')
    return None




#### PART 3 PROVIDED CODE ####
"""
Expects 3 lists - one of x labels, one of data labels, and one of data values - and a title.
You can use it to graph any number of datasets side-by-side to compare and contrast.
"""
def sideBySideBarPlots(xLabels, labelList, valueLists, title):
    import matplotlib.pyplot as plt

    w = 0.8 / len(labelList)  # the width of the bars
    xPositions = []
    for dataset in range(len(labelList)):
        xValues = []
        for i in range(len(xLabels)):
            xValues.append(i - 0.4 + w * (dataset + 0.5))
        xPositions.append(xValues)

    for index in range(len(valueLists)):
        plt.bar(xPositions[index], valueLists[index], width=w, label=labelList[index])

    plt.xticks(ticks=list(range(len(xLabels))), labels=xLabels, rotation="vertical")
    plt.legend()
    plt.title(title)

    plt.show()

"""
Expects two lists of probabilities and a list of labels (words) all the same length
and plots the probabilities of x and y, labels each point, and puts a title on top.
Expects that the y axis will be from -1 to 1. If you want a different y axis, change plt.ylim
"""
def scatterPlot(xValues, yValues, labels, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    plt.scatter(xValues, yValues)

    # make labels for the points
    for i in range(len(labels)):
        plt.annotate(labels[i], # this is the text
                    (xValues[i], yValues[i]), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0, 10), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center

    plt.title(title)
    plt.ylim(-1, 1)

    # a bit of advanced code to draw a line on y=0
    ax.plot([0, 1], [0.5, 0.5], color='black', transform=ax.transAxes)

    plt.show()


### RUN CODE ###

# This code runs the test cases to check your work
if __name__ == "__main__":
    # print("\n" + "#"*15 + " WEEK 1 TESTS " +  "#" * 16 + "\n")
    # test.week1Tests()
    # print("\n" + "#"*15 + " WEEK 1 OUTPUT " + "#" * 15 + "\n")
    # test.runWeek1()
    

    # ## Uncomment these for Week 2 ##
    # print("\n" + "#"*15 + " WEEK 2 TESTS " +  "#" * 16 + "\n")
    # test.week2Tests()
    # print("\n" + "#"*15 + " WEEK 2 OUTPUT " + "#" * 15 + "\n")
    # test.runWeek2()

    ## Uncomment these for Week 3 ##
    print("\n" + "#"*15 + " WEEK 3 OUTPUT " + "#" * 15 + "\n")
    test.runWeek3()
