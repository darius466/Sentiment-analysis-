from textblob import TextBlob
from sys import argv
import time
from benchmarker import Benchmarker

pos_para_sent = 0.0
neg_para_sent = 0.0
pos_para_sub = 0.0
neg_para_sub = 0.0
pos_counter = 0
neg_counter = 0

#ref globals inside function

def blob_test():
    global pos_para_sent
    global neg_para_sent
    global pos_para_sub
    global neg_para_sub
    global pos_counter
    global neg_counter

    #read positive and get sentiment and subjectivity
    with open("positive.txt", "r") as p:
        for i in p.read().split('\n'):
            pos_counter += 1
            blob = TextBlob(i)
            #para_sent += round(vader['compound'] / counter, 4)
            #para_sub += round(1 - abs(vader['compound']) / counter, 4)
            pos_para_sent += blob.sentiment.polarity
            pos_para_sub += blob.sentiment.subjectivity
    p.close()

    #read negative and get sentiment and subjectivity 
    with open("negative.txt", "r") as n:
        for i in n.read().split('\n'):
            neg_counter += 1
            blob = TextBlob(i)
            #para_sent += round(vader['compound'] / counter, 4)
            #para_sub += round(1 - abs(vader['compound']) / counter, 4)
            neg_para_sent += blob.sentiment.polarity
            neg_para_sub += blob.sentiment.subjectivity
    n.close()

    #print('The positive file subjectivity is ', round(pos_para_sub / pos_counter, 4), ' and the sentiment is ', round(pos_para_sent / pos_counter, 4), '.')
    #print('The negative file subjectivity is ', round(neg_para_sub / neg_counter, 4), ' and the sentiment is ', round(neg_para_sent / neg_counter, 4), '.')
    #print('Time elapsed: ',end - start,' seconds')

def blob_acc():
    pos_count = 0
    pos_correct = 0

    with open("positive.txt", "r") as p:
        for line in p.read().split('\n'):
            analysis = TextBlob(line)
            #measure accuracy if subjectivity meets a certain threshold 
            if analysis.sentiment.subjectivity >= 0.9:
                if analysis.sentiment.polarity > 0:
                    pos_correct += 1
                pos_count += 1
    p.close()

    neg_count = 0
    neg_correct = 0

    with open("negative.txt", "r") as n:
        for line in n.read().split('\n'):
            analysis = TextBlob(line)

            if analysis.sentiment.subjectivity > 0.9:
                if analysis.sentiment.polarity <= 0:
                    neg_correct += 1
                neg_count += 1
    n.close()

    #To measure textblob accuracy, we only want to account for those samples with high degrees of subjectivity
    print("pos accuracy: ",pos_correct/pos_count*100.0,"% via ",pos_count," examples.")
    print("neg accuracy: ",neg_correct/neg_count*100.0,"% via ",neg_count," examples.")

#cpu util
def blob_cpu():
    #specify number of loop
    with Benchmarker(1000*1000, width=20) as bench:
        s1 = blob_test()

        @bench(None)                ## empty loop
        def _(bm):
            for i in bm:
                pass

test_length = 1
total_time = 0.0

#run test
for i in range(0, test_length):
    start = time.time()
    blob_test()
    end = time.time()
    total_time += end - start

avg_time = round(total_time / test_length, 4)

print('The positive file subjectivity is ', round(pos_para_sub / pos_counter, 4), ' and the sentiment is ', round(pos_para_sent / pos_counter, 4), '.')
print('The negative file subjectivity is ', round(neg_para_sub / neg_counter, 4), ' and the sentiment is ', round(neg_para_sent / neg_counter, 4), '.')
print('average elapsed time: ',avg_time,' s')
print('')
blob_acc()
print('')    
blob_cpu()