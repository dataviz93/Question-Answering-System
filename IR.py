
# coding: utf-8

# In[88]:

#Read the file

f = open('SkypeFAQ.txt')
questions =[]
answers = []
text = f.read()
text = re.split('\n{2,}',text)
for line in f.readlines():
    if '?' in line:
        questions.append(line)
    else:
        if line != '\n': answers.append(line)

assert len(questions) == len(answers)


# In[150]:

#Read the file for O365

f = open('O365FAQ.txt')
questions =[]

question_line_indices = []
text = f.readlines()
for i,line in enumerate(text):
    if '?' in line:
        question_line_indices.append(i)
        questions.append(line)
print question_line_indices
answers = [''] * len(questions)
cnt = 0
flag = True
while flag:
    if cnt >= len(question_line_indices) :
        break
    answer_start = question_line_indices[cnt] + 1
    if cnt < len(question_line_indices)-1:
        answer_end = question_line_indices[cnt+1]
    else: answer_end = len(text)-1
    if answer_end > question_line_indices[-1]: break
    answers[cnt] = ' '.join(text[answer_start:answer_end])
    cnt = cnt + 1
    assert len(questions) == len(answers)



# In[179]:

#Clean and tokenize the sentences
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import string
all_text = [questions[i].decode('utf-8','replace') + ' ' + answers[i].decode('utf-8','replace') for i in range(0,len(questions))]

#Add synonyms from wordnet

all_text_cleaned = [sen.replace("-",' ') for sen in all_text]

stops = set(stopwords.words('english'))
punc = list(string.punctuation)
punc.remove("'")
punc.append('\n')
all_text_cleaned = [''.join(ch for ch in sen.lower() if ch not in punc) for sen in all_text_cleaned]

#Remove stop words
all_text_cleaned = [' '.join(word for word in sen.split() if word not in stops) for sen in all_text_cleaned]





# In[ ]:

#Add synonymous words into the mix.


# In[ ]:

#StLemmatize
Stemmer = PorterStemmer()
Lemmatizer = WordNetLemmatizer()
for i,sen in enumerate(all_text_cleaned):
    all_text_cleaned[i] = ' '.join([Stemmer.stem(Lemmatizer.lemmatize(word)) for word in sen.split()])


# In[176]:

#Vectorize each sentence
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(ngram_range = (1,3),min_df=1)

tfidf_train = vectorizer.fit_transform(all_text_cleaned)
vectorizer.get_feature_names()


# In[180]:

input_question = ["how many users on o365"]
input_question = [' '.join([Stemmer.stem(Lemmatizer.lemmatize(word)) for word in input_question[0].split()])]
tfidf_test_vector = vectorizer.transform(input_question)
print(tfidf_train.shape,tfidf_test_vector.shape)

sim = tfidf_train * tfidf_test_vector.T
import operator
max_index, max_value = max(enumerate(sim), key=operator.itemgetter(1))

print sim
if max_value >= 0.01:
    print(all_text[index])
else: print("There is no suitable answer to match your question.")
    

