"""
15-110 Hw6 - Language Modeling Project
Name: M.Bharadwaj
AndrewID: 2023501046
"""

import hw6_language_tests as test

project = "Language" # don't edit this

### WEEK 1 ###

'''
loadBook(filename)
#1 [Check6-1]
Parameters: str
Returns: 2D list of strs
'''
def loadBook(filename):                       # This function takes the book as filename and converts into 2D list called as corpus.
    corpus = []
    f = open(filename,"r")                    # Opens the file and reads it.
    # reader = f.read()
    # lines = f.split()
    for i in f:
        lines = i.split()
        # print(lines)
        if lines:
            corpus.append(lines)
    # print(corpus)
    return corpus


'''
getCorpusLength(corpus)
#2 [Check6-1]
Parameters: 2D list of strs
Returns: int
'''
def getCorpusLength(corpus):                    # This function is used to find the count of total number of unigrams in the 2D list.
    corpus_count = 0
    for i in corpus:
        # print(i)
        corpus_count += len(i)
    return corpus_count


'''
buildVocabulary(corpus)
#3 [Check6-1]
Parameters: 2D list of strs
Returns: list of strs
'''
def buildVocabulary(corpus):                    # This function is to find the unique unigrams in the corpus 2D list.
    unique_unigrams = []
    for i in corpus:
        for j in i:
            if j not in unique_unigrams:
                unique_unigrams.append(j)
    return unique_unigrams


'''
countUnigrams(corpus)
#4 [Check6-1]
Parameters: 2D list of strs
Returns: dict mapping strs to ints
'''
def countUnigrams(corpus):                           # This function is used to organize the data.
    unigram_dictionary = {}                          # Here we have created a dictionary where the keys are unigrams in corpus and values are how many times each unigram occured.
    for i in corpus:
        for j in i:
            if j not in unigram_dictionary:
                unigram_dictionary[j] = 1
            else:
                unigram_dictionary[j] += 1
    return unigram_dictionary


'''
getStartWords(corpus)
#5 [Check6-1]
Parameters: 2D list of strs
Returns: list of strs
'''
def getStartWords(corpus):                                   # This function helps to keep track on start words that is words that start sentences.                                     
    unique_start_words = []                                  # This list contains words that start the sentences and unique.
    for i in corpus:
        for j in i:
            if j == i[0] and j not in unique_start_words:
                unique_start_words.append(j)
    return unique_start_words


'''
countStartWords(corpus)
#5 [Check6-1]
Parameters: 2D list of strs
Returns: dict mapping strs to ints
'''
def countStartWords(corpus):                             # This function returns dictionary that maps starts words to how many times each start word appeared and count of start word also.
    startword_count = {}
    for i in corpus:
        # for j in i:
        if i[0] not in startword_count:
            startword_count[i[0]] = 1
        else:
            startword_count[i[0]] += 1

    return startword_count


'''
countBigrams(corpus)
#6 [Check6-1]
Parameters: 2D list of strs
Returns: dict mapping strs to (dicts mapping strs to ints)
'''
def countBigrams(corpus):                                 # This function is used to build bigrams they are pair of words that appear next to each other.
    unique_biagram = {}                                   # This function returns a 2D dictionary with count of each unique bigram in corpus.
    for i in corpus:
        for j in range(len(i)-1):
            if i[j] not in unique_biagram:
                unique_biagram[i[j]] = {}
            if i[j+1] not in unique_biagram[i[j]]:
                unique_biagram[i[j]][i[j+1]] = 1
            else:
                unique_biagram[i[j]][i[j+1]] += 1 
    return unique_biagram


### WEEK 2 ###

'''
buildUniformProbs(unigrams)
#1 [Check6-2]
Parameters: list of strs
Returns: list of floats
'''
def buildUniformProbs(unigrams):                       # This function is used to create a uniform model measn a model where all words get the probability of 1/N, N is size of vocalbulary.                                                                                         
                                                       # Here we used buildvocabulary() function to compute list of all words.
    unigram_probability = []
    for i in range(len(unigrams)):
        unigram_probability.append(1/len(unigrams))     # The new list contanins of same length where 1/len(unigram) is the value at each index.
    return unigram_probability


'''
buildUnigramProbs(unigrams, unigramCounts, totalCount)
#2 [Check6-2]
Parameters: list of strs ; dict mapping strs to ints ; int
Returns: list of floats
'''
def buildUnigramProbs(unigrams, unigramCounts, totalCount):                      # This function is used to predict the words in the text by how frequently they occur.
    probability_words = []                                                       # Here we will return a list of the probabilities of each word. To do this it takes list of unique unigrams,
    for i in range(len(unigrams)):                                               # and a dict mapping unique unigrams to counts and total count of words in the book. 
        probability_words.append(unigramCounts[unigrams[i]]/totalCount)
    return probability_words


'''
buildBigramProbs(unigramCounts, bigramCounts)
#3 [Check6-2]
Parameters: dict mapping strs to ints ; dict mapping strs to (dicts mapping strs to ints)
Returns: dict mapping strs to (dicts mapping strs to (lists of values))
'''
def buildBigramProbs(unigramCounts, bigramCounts):                                        # This function is used to predict the probability of one word changes based on the word before it.
    nested_dict={}                                                                        # Here this function returns a new nested dictionary in which each key is a word in vocabulary and values are bigram probability dictionary for that word.
    # words = []                                                                          
    # probability = []
    for prevWord in bigramCounts:
        prev_words = []
        probability = []
        for word in bigramCounts[prevWord]:
            probability.append(bigramCounts[prevWord][word]/unigramCounts[prevWord])
            prev_words.append(word)
        nested_dict[prevWord]={}
        nested_dict[prevWord]["words"] = prev_words
        nested_dict[prevWord]["probs"] = probability
    return nested_dict


'''
getTopWords(count, words, probs, ignoreList)
#4 [Check6-2]
Parameters: int ; list of strs ; list of floats ; list of strs
Returns: dict mapping strs to floats
'''
def getTopWords(count, words, probs, ignoreList):                                        # This function is to find the top count of words with highest probabilities, and returns a dictionary with keys as high probability words and values as that word probability.
    highest_prob = {}
    while count != len(highest_prob):                                                    # To do that this function takes no.of words, a list of words, a list of corresponding probabilities and a ignored words list.
        i = probs.index(max(probs))
        if words[i] not in ignoreList and words[i] not in highest_prob:
            highest_prob[words[i]] = probs[i]
        probs.remove(probs[i])
        words.remove(words[i])
    return highest_prob


'''
generateTextFromUnigrams(count, words, probs)
#5 [Check6-2]
Parameters: int ; list of strs ; list of floats
Returns: str
'''
from random import choices
def generateTextFromUnigrams(count, words, probs):                        # This function returns a string generated by concatenating probabilistically choosen words together.
    text = ""                                    
    for i in range(count):
        word = choices(words,weights = probs)                             # Here we use choices() function from the random library to generate random words. This function returns a list conataining the word it picked.
        text += word[0] + " "
        text.strip()
    return text


'''
generateTextFromBigrams(count, startWords, startWordProbs, bigramProbs)
#6 [Check6-2]
Parameters: int ; list of strs ; list of floats ; dict mapping strs to (dicts mapping strs to (lists of values))
Returns: str
'''
def generateTextFromBigrams(count, startWords, startWordProbs, bigramProbs):                                        # This function is used to generate a string of count words generated probabilistically.
    generated_sent = ""
    words_lst = []                                                                                                  
    while count != len(words_lst):
        if generated_sent == "" or words_lst[-1] == ".":
            word = choices(startWords,weights=startWordProbs)
            words_lst.append(word[0])
            generated_sent += word[0] + " "
        else:
            if words_lst[-1] in bigramProbs:
                word = choices(bigramProbs[words_lst[-1]]["words"],weights=bigramProbs[words_lst[-1]]["probs"])
            words_lst.append(word[0])
            generated_sent += word[0] + " "
    return generated_sent


### WEEK 3 ###

ignore = [ ",", ".", "?", "'", '"', "-", "!", ":", ";", "by", "around", "over",
           "a", "on", "be", "in", "the", "is", "on", "and", "to", "of", "it",
           "as", "an", "but", "at", "if", "so", "was", "were", "for", "this",
           "that", "onto", "from", "not", "into" ]

'''
graphTop50Words(corpus)
#3 [Hw6]
Parameters: 2D list of strs
Returns: None
'''
def graphTop50Words(corpus):                                                                 # This function is used to compute top 50 most frequent words according to probabilities and returns bar graph.
    corpus_count = getCorpusLength(corpus)
    unigram = buildVocabulary(corpus)                                                        # Done by using the dictionaries generated in buildVocabulary(), getCorpusLenght(), countUnigrams() and top 50 words from getTopWords(). To generate bar graph we use the function barPlot().
    unigram_count = countUnigrams(corpus)
    unigram_probability = buildUnigramProbs(unigram,unigram_count,corpus_count)
    dict = getTopWords(50,unigram,unigram_probability,ignore)
    
    return barPlot(dict,"Graph of top 50 most common words")


'''
graphTopStartWords(corpus)
#4 [Hw6]
Parameters: 2D list of strs
Returns: None
'''
def graphTopStartWords(corpus):                                                        # This function takes corpus and previous functions to compute the start words and their probabilities and returns bar grpah.
    startingWord_probability = []
    starting_word = getStartWords(corpus)
    startingWord_count = countStartWords(corpus)                                       # Done using start word dictionary generatedn in the getTopWords(), getStartWordsCount() and getStartWords() functions and bar grapgh is generated uisng the barPlat() function.
    word_count = sum(list(startingWord_count.values()))
    for i in startingWord_count:
        startingWord_probability.append(startingWord_count[i]/word_count)
    word_dict = getTopWords(50,starting_word,startingWord_probability,ignore)
   
    return barPlot(word_dict,"Graph of top 50 starting words")


'''
graphTopNextWords(corpus, word)
#5 [Hw6]
Parameters: 2D list of strs ; str
Returns: None
'''
def graphTopNextWords(corpus, word):                                                                                                              # This function takes corpus and words and returns the graph of top 10 words that appear after the word in corpus.
    countUnigram = countUnigrams(corpus)                                                                                                          # Done by taking the dictioanries generated in countUnigrams() and countBigrams() and the top 10 from getTopWords(). The bar graph is generated using barPlot() function.
    countBigram = countBigrams(corpus)
    BigramProbability = buildBigramProbs(countUnigram,countBigram)

    return barPlot(getTopWords(10,BigramProbability[word]["words"],BigramProbability[word]["probs"],ignore),"Graph of top next Words")


'''
setupChartData(corpus1, corpus2, topWordCount)
#6 [Hw6]
Parameters: 2D list of strs ; 2D list of strs ; int
Returns: dict mapping strs to (lists of values)
'''
def setupChartData(corpus1, corpus2, topWordCount):                                   # This function is used to compare probabilities of the most common words in two books.
    top_dict1={}
    top_dict2={}
    top_dict3={}
    corpus_count1=getCorpusLength(corpus1)                                           # This is done by taking two corpus list from two books, a number of top words count from two corpuses.
    corpus_count2=getCorpusLength(corpus2)                                           # And also considering the probabilities of all top words in two corpuses, and bigrams and their probabilities.
    unigram1=buildVocabulary(corpus1)
    unigram2=buildVocabulary(corpus2)
    unigramCount1=countUnigrams(corpus1)
    unigramCount2=countUnigrams(corpus2)                                             # This function returns a dictioanry of three list they are top words and the two probability lists.
    unigramProbs1=buildUnigramProbs(unigram1,unigramCount1,corpus_count1)
    unigramProbs2=buildUnigramProbs(unigram2,unigramCount2,corpus_count2)
    top_dict1=getTopWords(topWordCount,unigram1,unigramProbs1,ignore)
    top_dict2=getTopWords(topWordCount,unigram2,unigramProbs2,ignore)
    for i in top_dict2:       
        if i not in top_dict1:
            top_dict1[i]=0
    
    for i in top_dict1:
        if i not in top_dict2:
            top_dict3[i]=0
        else:top_dict3[i]=top_dict2[i]
    totalUnigrams=list(top_dict1.keys())
    count_1_probs=list(top_dict1.values())
    count_2_probs=list(top_dict3.values())

    return {"topWords":totalUnigrams, "corpus1Probs":count_1_probs, "corpus2Probs":count_2_probs}


'''
graphTopWordsSideBySide(corpus1, name1, corpus2, name2, numWords, title)
#6 [Hw6]
Parameters: 2D list of strs ; str ; 2D list of strs ; str ; int ; str
Returns: None
'''
def graphTopWordsSideBySide(corpus1, name1, corpus2, name2, numWords, title):                                # This function is used to generate the side by side bar graph for the two corpus lists by taking two corpuses, the no.of words, and a title.
    c = setupChartData(corpus1,corpus2,numWords)
    sideBySideBarPlots(c["topWords"], c["corpus1Probs"], c["corpus2Probs"], name1, name2, title)             # The function sideBYSideBarPlots() function is used to generate side by side bar grpah.
    return None


'''
graphTopWordsInScatterplot(corpus1, corpus2, numWords, title)
#6 [Hw6]
Parameters: 2D list of strs ; 2D list of strs ; int ; str
Returns: None
'''
def graphTopWordsInScatterplot(corpus1, corpus2, numWords, title):                 # This function is used to generate scatterd plot graph by using two corpus lists and their probabilities.       
    c = setupChartData(corpus1,corpus2,numWords)
    scatterPlot(c["corpus1Probs"], c["corpus2Probs"], c["topWords"], title)        # The function scatterPlot() is used to create the scatter plot graph.
    return None


### WEEK 3 PROVIDED CODE ###

"""
Expects a dictionary of words as keys with probabilities as values, and a title
Plots the words on the x axis, probabilities as the y axis and puts a title on top.
"""
def barPlot(dict, title):
    import matplotlib.pyplot as plt

    names = []
    values = []
    for k in dict:
        names.append(k)
        values.append(dict[k])

    plt.bar(names, values)

    plt.xticks(rotation='vertical')
    plt.title(title)

    plt.show()

"""
Expects 3 lists - one of x values, and two of values such that the index of a name
corresponds to a value at the same index in both lists. Category1 and Category2
are the labels for the different colors in the graph. For example, you may use
it to graph two categories of probabilities side by side to look at the differences.
"""
def sideBySideBarPlots(xValues, values1, values2, category1, category2, title):
    import matplotlib.pyplot as plt

    w = 0.35  # the width of the bars

    plt.bar(xValues, values1, width=-w, align='edge', label=category1)
    plt.bar(xValues, values2, width= w, align='edge', label=category2)

    plt.xticks(rotation="vertical")
    plt.legend()
    plt.title(title)

    plt.show()

"""
Expects two lists of probabilities and a list of labels (words) all the same length
and plots the probabilities of x and y, labels each point, and puts a title on top.
Note that this limits the graph to go from 0x0 to 0.02 x 0.02.
"""
def scatterPlot(xs, ys, labels, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    plt.scatter(xs, ys)

    # make labels for the points
    for i in range(len(labels)):
        plt.annotate(labels[i], # this is the text
                    (xs[i], ys[i]), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0, 10), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center

    plt.title(title)
    plt.xlim(0, 0.02)
    plt.ylim(0, 0.02)

    # a bit of advanced code to draw a y=x line
    ax.plot([0, 1], [0, 1], color='black', transform=ax.transAxes)

    plt.show()


### RUN CODE ###

# This code runs the test cases to check your work
if __name__ == "__main__":
    print("\n" + "#"*15 + " WEEK 1 TESTS " +  "#" * 16 + "\n")
    test.week1Tests()
    # test.testLoadBook()
    # test.testGetCorpusLength()
    # test.testBuildVocabulary()
    # test.testCountUnigrams()
    # test.testGetStartWords()
    # test.testCountStartWords()
    # test.testCountStartWords()
    # test.testCountBigrams()
    print("\n" + "#"*15 + " WEEK 1 OUTPUT " + "#" * 15 + "\n")
    test.runWeek1()

    ## Uncomment these for Week 2 ##

    print("\n" + "#"*15 + " WEEK 2 TESTS " +  "#" * 16 + "\n")
    test.week2Tests()
    # test.testBuildUniformProbs()
    # test.testBuildUnigramProbs()
    # test.testBuildBigramProbs()
    # test.testGetTopWords()
    # test.testGenerateTextFromUnigrams()
    # test.testGenerateTextFromBigrams()
    print("\n" + "#"*15 + " WEEK 2 OUTPUT " + "#" * 15 + "\n")
    test.runWeek2()


    ## Uncomment these for Week 3 ##

    print("\n" + "#"*15 + " WEEK 3 OUTPUT " + "#" * 15 + "\n")
    # test.testSetupChartData()
    test.runWeek3()