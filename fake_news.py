import numpy as np
import nltk
from nltk.corpus import stopwords
import pandas as pd 
import math
import operator
import plotly
import plotly.graph_objs as go
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score 
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import normalize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nltk.download('stopwords')

_  = np.seterr(divide='ignore', invalid='ignore')
plotly.offline.init_notebook_mode(connected=True)

training_data = pd.read_csv("./train.csv")
test_data = pd.read_csv("./test.csv")
submit_data = pd.read_csv("./countvect5.csv")

training_data = training_data.fillna(' ')
test_data = test_data.fillna(' ')

print(training_data.columns)
print("Len training data: %i" % (len(training_data)))
print(training_data[:10])


print("Len test data: %i" % (len(test_data)))
print(test_data[:10])


print("Training data :")
print('--------------------------------------------------')
print('Total number of articles: ',len(training_data))
print('Number of fake news: ',len(training_data[training_data.label==1]))
training_data[training_data.label==1].head()


print(training_data.label.value_counts())
print("Of a total: " + str(len(training_data)))
print('--------------------------------------------------')
print('Percentage of unreliable news:   ', str(len(training_data[training_data.label==1])/len(training_data)))
print('Percentage of reliable news:   ', str(len(training_data[training_data.label==0])/len(training_data)))
training_data.label.value_counts().plot(kind='pie', title = "Percentage of reliable vs unreliable news", label = 'type of news')


training_dict_fake = {}
training_dict_real = {}
for index, row in training_data.iterrows():
    if(row['author'])!= ' ':
        if(row['label'] == 1):
            auth=row['author']
            if auth in training_dict_fake:
                training_dict_fake[auth] = training_dict_fake[auth]+1
            else:
                training_dict_fake[auth] = 1
        elif(row['label'] == 0):
            auth=row['author']
            if auth in training_dict_real:
                training_dict_real[auth] = training_dict_real[auth]+1
            else:
                training_dict_real[auth] = 1

print('100 Most Active Authors of Unreliable News')
sorted_real_top100 = sorted(training_dict_real.items(), key=operator.itemgetter(1), reverse=True)[:100]
print(sorted_real_top100)
print('\n--------------------------------------------------\n')
print('100 Most Active Authors of Reliable News')
sorted_fake_top100 = sorted(training_dict_fake.items(), key=operator.itemgetter(1), reverse=True)[:100]
print(sorted_fake_top100)


print('Authors from the Top 100 who write both Reliable and Unreliable News')
common_authors_top100 = list(set(sorted_real_top100) & set(sorted_fake_top100))
print(common_authors_top100)


sorted_real_all = sorted(training_dict_real.items(), key=operator.itemgetter(1), reverse=True)
# print(sorted_real)
sorted_fake_all = sorted(training_dict_fake.items(), key=operator.itemgetter(1), reverse=True)
# print(sorted_fake)

print('All Authors writing Unreliable News')
sorted_real_authors_all = []
for w in sorted_real_all:
    sorted_real_authors_all.append(w[0])

print(sorted_real_authors_all)

print('\n--------------------------------------------------\n')
print('All Authors writing Reliable News')
sorted_fake_authors_all = []
for w in sorted_fake_all:
    sorted_fake_authors_all.append(w[0])

print(sorted_fake_authors_all)


print('All authors who write both Reliable and Unreliable News (*from dataset)')
common_authors_all = list(set(sorted_real_authors_all) & set(sorted_fake_authors_all))
print(common_authors_all)
print('\n--------------------------------------------------\n')
print('100 Most Active Authors of Reliable News')
print('# of authors writing both fake and real news:', str(len(common_authors_all)))
print('Of total:', str(len(sorted_real_authors_all)+len(sorted_fake_authors_all)-len(common_authors_all)))

print('Percentage of authors writing both fake and real news:', str(len(common_authors_all)/(len(sorted_real_authors_all)+len(sorted_fake_authors_all)-len(common_authors_all))))


stop_words = list(set(stopwords.words('english')))
stop_words.append('-')
stop_words.append('|')
stop_words.append('&')
stop_words.append('–')

training_dict_fake = {}
training_dict_real = {}
real_word_count = 0
fake_word_count = 0
for index, row in training_data.iterrows():
    if(row['title']):
        for word in row['title'].split():
            x = word.lower()
            if x not in stop_words:
                if(row['label'] == 1):
                    fake_word_count += 1
                    if word in training_dict_fake:
                        training_dict_fake[word] = training_dict_fake[word]+1
                    else:
                        training_dict_fake[word] = 1
                else:
                    real_word_count += 1
                    if word in training_dict_real:
                        training_dict_real[word] = training_dict_real[word]+1
                    else:
                        training_dict_real[word] = 1

sorted_real = sorted(training_dict_real.items(), key=operator.itemgetter(1), reverse=True)
sorted_fake = sorted(training_dict_fake.items(), key=operator.itemgetter(1), reverse=True)


sorted_real_words = []
for w in sorted_real:
    sorted_real_words.append(w[0])

# print(sorted_real_words)
print("Number of real words: ", real_word_count)
print("Number of fake words: ", fake_word_count)


sorted_fake_words = []
for w in sorted_fake:
    sorted_fake_words.append(w[0])

print(sorted_fake_words)


common_words = list(set(sorted_real_words) & set(sorted_fake_words))

print("These are common words found in both reliable and unreliable news articles.") 
print()
print(common_words)

class Words(object):
        
    def count(self, w) :
        return self.all[w]
        
    def noncount(self, w) :
        return self.len - self.count(w)
    
    def rate(self, w) :
        return self.count(w) / self.len
    
    def counts(self, w) :
        return [c[w] for c in self.cts]
    
    def noncounts(self, w) :
        return [self.tot[n] - c[w] for n,c in enumerate(self.cts)]
    
    def rates(self, w, include_zeros=True) :
        return [c[w] / self.tot[n] 
                for (n,c) in enumerate(self.cts)
                if c[w] != 0 or include_zeros]
    
    def interval(self, w) :
        r = self.rates(w)
        return max(r) - min(r)
    
    def matrix(self, features) :
        result = np.zeros((len(self.cts), len(features)))
        for n, w in enumerate(features):
            result[:,n] = self.rates(w)
        return normalize(result, norm='l2')

analyser = SentimentIntensityAnalyzer()

def get_sentiment(w):
    score = analyser.polarity_scores(w)
    return score['pos'], score['neg'], score['compound']


titles = [x for x in training_data.title.tolist() if x is not None]


def text_description(a1, a2, w) :
    list_of_titles = random.choice([i for i in titles if w in i.split()])
    pos, neg, obj = get_sentiment(w)
    return u"{}: Pos: {} Neg: {} Example title:{}".format(
        w, pos, neg, list_of_titles)

def ll_diff_score(a1, a2, w) :
    return compute_ll(a1[w], a2[w], real_word_count-a1[w], fake_word_count-a2[w])

def ll_diff_scores(a1, a2, w) :
    results = {w: ll_diff_score(a1, a2, w) for w in common_words}
    return results

def compute_ll(a, b, c, d) :
    rate = (a+b) / (c+d)
    e1 = c * rate
    e2 = d * rate
    ll = 2 * (a * math.log(a/e1) + b * math.log(b/e2))
    return ll

def visual_size(a1, a2, w):
    return 10


diffs = ll_diff_scores(training_dict_real, training_dict_fake, common_words)
def llcontrastscatter(a1, a2, llvs) :
    x = [(a1[w]/real_word_count) for w in llvs]
    y = [(a2[w]/fake_word_count) for w in llvs]
    text = [text_description(a1, a2, w) for w in llvs]
#     color = [math.pow(llvs[w], -1/10) for w in llvs]
    color = []
    for w in llvs:
        pos, neg, compound = get_sentiment(w.lower())
        if(pos == 0 and neg == 0):
            color.append(0)
        else:
            color.append(compound*50)
    size = [visual_size(a1, a2, w) for w in llvs]
    return dict(x=x, y=y, text=text, 
                marker=dict(size=size, 
                            color=color, 
                            colorscale='viridis'), 
                mode='markers', hoverinfo='text')

plotly.offline.iplot(dict(data=[go.Scatter(llcontrastscatter(training_dict_real, training_dict_fake, diffs))],
                          layout=dict(title=u'Real vs Fake',
                                      hovermode='closest',
                                      shapes=[dict(type='line',x0=1e-5,y0=1e-5,x1=5e-2,y1=5e-2),
                                             dict(type='line',x0=2e-5,y0=1e-5,x1=5e-2,y1=2.5e-2,line=dict(dash='dot')),
                                             dict(type='line',x0=1e-5,y0=2e-5,x1=5e-2,y1=1e-1,line=dict(dash='dot'))],
                                      xaxis=dict(type='log', title=u'Frequency in Reliable'),
                                      yaxis=dict(type='log', title=u'Frequency in Unreliable'))))


fake_news_sentiment = {'pos':0, "neg":0, 'compound':0}
real_news_sentiment = {'pos':0, "neg":0, 'compound':0}

fake_news_capital = 0
real_news_capital = 0

fake_news_len = 0
real_news_len = 0

for index, row in training_data.iterrows():
    if(row['title']):
        pos, neg, compound = get_sentiment(row['title'])
        if(row['label'] == 1):
            fake_news_len += len(row['title'])
            fake_news_capital += sum(1 for c in row['title'].split() if c.isupper())
            
            fake_news_sentiment['pos'] = fake_news_sentiment['pos'] + pos
            fake_news_sentiment['neg'] = fake_news_sentiment['neg'] + neg
            fake_news_sentiment['compound'] = fake_news_sentiment['compound'] + compound
        else:
            real_news_len += len(row['title'])
            real_news_capital += sum(1 for c in row['title'].split() if c.isupper())

            real_news_sentiment['pos'] = real_news_sentiment['pos'] + pos
            real_news_sentiment['neg'] = real_news_sentiment['neg'] + neg
            real_news_sentiment['compound'] = real_news_sentiment['compound'] + compound

print("Fake news: sentiment: {} \ntitle length: {} capital: {}".format(fake_news_sentiment, fake_news_len, fake_news_capital))
print("Real news: sentiment: {} \ntitle length: {} capital: {}".format(real_news_sentiment, real_news_len, real_news_capital))

# Fake news capitalizes words a lot more, as a percentage


training_data['article'] = training_data['title'] + training_data['author'] + training_data['text']
test_data['article'] = test_data['title'] + test_data['author'] + test_data['text']

print(training_data)


tfidf_vectorizer = TfidfVectorizer(stop_words='english')


x_train = tfidf_vectorizer.fit_transform(training_data['title'].values)
x_test = tfidf_vectorizer.transform(test_data['title'].values)
y_train = training_data['label']
y_test = submit_data['label']

model = MultinomialNB()
model.fit(x_train, y_train)

accuracies = cross_val_score(estimator=model, X=x_train, y=y_train, cv=10)
mean = accuracies.mean()
standard_deviation = accuracies.std()

print('Mean accuracy on training data: ', mean)
print('Standard deviation: ',standard_deviation)


predictions = model.predict(x_test)
print("Accuracy on test data: ", accuracy_score(y_test, predictions, normalize=True))
print(confusion_matrix(y_test, predictions))


x_train_auth = tfidf_vectorizer.fit_transform(training_data['author'].values)
x_test_auth = tfidf_vectorizer.transform(test_data['author'].values)
y_train_auth = training_data['label']
y_test_auth = submit_data['label']

model_auth = MultinomialNB()
model_auth.fit(x_train_auth, y_train_auth)

accuracies_auth = cross_val_score(estimator=model_auth, X=x_train_auth, y=y_train_auth, cv=10)
mean_auth = accuracies_auth.mean()
standard_deviation_auth = accuracies_auth.std()

print('Mean accuracy on training data: ', mean_auth)
print('Standard deviation: ',standard_deviation_auth)

predictions_auth = model_auth.predict(x_test_auth)
print("Accuracy on test data: ", accuracy_score(y_test_auth, predictions_auth, normalize=True))
print(confusion_matrix(y_test_auth, predictions_auth))

results = pd.DataFrame({'id':test_data['id'],'title':test_data['title'],'author':test_data['author'],'text':test_data['text'],'label':predictions})
print(results)