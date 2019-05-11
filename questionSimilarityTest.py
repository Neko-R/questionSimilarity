#%% [markdown]
#The Start Of Something New

#%%
import pandas as pd
import numpy as np 
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import datetime

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stop_words.discard("for")
stop_words.discard("in")
stop_words.discard("while")
stop_words.discard("not")
stop_words.discard("if")

#print(doc2vec.FAST_VERSION)

#%%

df = pd.read_csv('questionsData.csv', encoding = "ISO-8859-1", usecols=['Id','Title', 'Body'])
#df = pd.read_csv('Questions.csv', encoding = "ISO-8859-1", usecols=['Id','Title'])

print(df.shape)

#%%
questions = (df['Title']).values.tolist() #d2v.model

def preProcess(sentence):
    tokens = word_tokenize(sentence)
    words = [w for w in tokens if ((w not in stop_words) and (w.isalpha()))]
    return words

tagged_data = [TaggedDocument(words=preProcess(question.lower()), tags=[i]) for i, question in enumerate(questions)]

#%%
model = Doc2Vec(vector_size=100, alpha=0.025, min_alpha=0.00025, window = 5, min_count=10, dm=1, workers=5)
  
model.build_vocab(tagged_data)

for epoch in range(50):
    print('iteration {0} at {1}'.format(epoch, datetime.datetime.now().time()))
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    model.alpha -= 0.0002
    model.min_alpha = model.alpha

model.save("d2v.model") #change
print("Model Saved")

#%%
postmodel = Doc2Vec.load("d2v.model")

text = "I want to know how to delete a character from a string".lower()
test_data = preProcess(text)
v1 = postmodel.infer_vector(test_data)

print("Question - {0}\n".format(text))
mostSim = postmodel.docvecs.most_similar(positive=[v1])
for m in mostSim:
    print(questions[m[0]], m[0])
    print(m[1])
    print("")

#%%
print("list:")
print(postmodel.wv.most_similar("list"))
print("")

print("array:")
print(postmodel.wv.most_similar("array"))
print("")

print("index:")
print(postmodel.wv.most_similar("index"))

#%%
vocab = postmodel.wv.vocab

text = "I want to know how to delete a character from a string".lower()
test_data = preProcess(text)

text2 = "Delete list of elements from a list".lower()
test_data2 = preProcess(text2)

text3 = "how to insert a character into string".lower()
test_data3 = preProcess(text3)

text4 = "Python: Delete a character from a string".lower()
test_data4 = preProcess(text4)

text5 = "How to delete a character from a string using python?".lower()
test_data5 = preProcess(text5)

text6 = "what is a variable".lower()
test_data6 = preProcess(text6)

text7 = "how to remove a character from a string".lower()
test_data7 = preProcess(text7)

print("{0}\n\n{1}\n\n{2}\n\n{3}\n\n{4}\n\n{5}\n".format(text, text2, text3, text4, text5, text6))

print("Model wv.n_similarity: ", postmodel.wv.n_similarity(test_data2,test_data))
print("Model wn.n_similarity: ", postmodel.wv.n_similarity(test_data3,test_data))
print("Model wn.n_similarity: ", postmodel.wv.n_similarity(test_data4,test_data))
print("Model wn.n_similarity: ", postmodel.wv.n_similarity(test_data5,test_data))
print("Model wn.n_similarity: ", postmodel.wv.n_similarity(test_data6,test_data))
print("")

v1 = postmodel.infer_vector(test_data, epochs=500)
v2 = postmodel.infer_vector(test_data2, epochs=500)
v3 = postmodel.infer_vector(test_data3, epochs=500)
v4 = postmodel.infer_vector(test_data4, epochs=500)
v5 = postmodel.infer_vector(test_data5, epochs=500)
v6 = postmodel.infer_vector(test_data6, epochs=500)
v7 = postmodel.infer_vector(test_data7, epochs=500)

vec1 = np.array(v1)
vec2 = np.array([v1, v2,v3,v4,v5,v6,v7])

print(postmodel.wv.cosine_similarities(vec1, vec2))

postmodel.wv.most_similar_to_given(test_data, [test_data2, test_data])

#%%
# test_data = [w for w in word_tokenize("how exactly can i go into a string and remove a character".lower()) if w not in stop_words]
# test_data2 = [w for w in word_tokenize("How to delete a character from a string using python?".lower()) if w not in stop_words]
# test_data3 = [w for w in word_tokenize("Remove or Replace a character in a string".lower()) if w not in stop_words]

#model.wv.most_similar("list")
#print(questions1[16195])

#print("V1_infer", v1)

# to find most similar doc using tags
#similar_doc = model.docvecs.most_similar('1')
#print(similar_doc)


# to find vector of doc in training data using tags or in other words, printing the vector of document at index 1 in training data
#print(model.docvecs['1'])
