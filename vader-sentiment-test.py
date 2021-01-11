from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from sys import argv
import time
from benchmarker import Benchmarker

analyzer = SentimentIntensityAnalyzer()
pos_para_sent = 0.0
neg_para_sent = 0.0
pos_para_sub = 0.0
neg_para_sub = 0.0
pos_counter = 0
neg_counter = 0

#ref globals inside function


def vader_sent():
    global analyzer
    global pos_para_sent
    global neg_para_sent
    global pos_para_sub 
    global neg_para_sub 
    global pos_counter 
    global neg_counter
    
    #vader sentiment scores via compound score
    #vader subjectivity is the absolute value of inverse
    with open("positive.txt", "r") as p:
        for i in p.read().split('\n'):
            pos_counter += 1
            vader = analyzer.polarity_scores(i)
            #para_sent += round(vader['compound'] / counter, 4)
            #para_sub += round(1 - abs(vader['compound']) / counter, 4)
            pos_para_sent += vader['compound']
            pos_para_sub += 1 - abs(vader['compound'])
    p.close()

    with open("negative.txt", "r") as n:
        for i in n.read().split('\n'):
            neg_counter += 1
            vader = analyzer.polarity_scores(i)
            #para_sent += round(vader['compound'] / counter, 4)
            #para_sub += round(1 - abs(vader['compound']) / counter, 4)
            neg_para_sent += vader['compound']
            neg_para_sub += 1 - abs(vader['compound'])
    n.close()

    #print('The positive file subjectivity is ', round(pos_para_sub / pos_counter, 4), ' and the sentiment is ', round(pos_para_sent / pos_counter, 4), '.')
    #print('The negative file subjectivity is ', round(neg_para_sub / neg_counter, 4), ' and the sentiment is ', round(neg_para_sent / neg_counter, 4), '.')
    #print('Time elapsed: ',end - start,' seconds')

def vad_acc():
    pos_count = 0
    pos_correct = 0

    #measure accuracy
    with open("positive.txt", "r") as p:
        for line in p.read().split('\n'):
            vader = analyzer.polarity_scores(line)
            if not vader['neg'] > 0.1:
                if vader['pos'] - vader['neg'] > 0:
                    pos_correct += 1
                pos_count += 1
    p.close()

    neg_count = 0
    neg_correct = 0

    with open("negative.txt", "r") as n:
        for line in n.read().split('\n'):
            vader = analyzer.polarity_scores(line)
            if not vader['pos'] > 0.1:
                if vader['pos'] - vader['neg'] <= 0:
                    neg_correct += 1
                neg_count += 1
    n.close()

    #computed by summing the valence scores of each word in lexicon, adjusted according to the rules, then normalized
    #to measure vader accuracy for positives, we classify as positive only if 'neg' is less than 0.1 and vice versa
    print("pos accuracy: ",pos_correct/pos_count*100.0,"% via ",pos_count," examples.")
    print("neg accuracy: ",neg_correct/neg_count*100.0,"% via ",neg_count," examples.")   

#cpu util
def vader_cpu():
    #specify number of loop
    with Benchmarker(1000*1000, width=20) as bench:
        s1 = vader_sent()

        @bench(None)                ## empty loop
        def _(bm):
            for i in bm:
                pass   
    
test_length = 1
total_time = 0.0

#run test
for i in range(0, test_length):
    start = time.time()
    vader_sent()
    end = time.time()
    total_time += end - start

avg_time = round(total_time / test_length, 4)

print('The positive file subjectivity is ', round(pos_para_sub / pos_counter, 4), ' and the sentiment is ', round(pos_para_sent / pos_counter, 4), '.')
print('The negative file subjectivity is ', round(neg_para_sub / neg_counter, 4), ' and the sentiment is ', round(neg_para_sent / neg_counter, 4), '.')
print('average elapsed time: ',avg_time,' s')
print('')
vad_acc()
print('')    
vader_cpu()