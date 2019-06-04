import numpy as np
import scipy.io as sio
from sklearn import svm
import re
from nltk import PorterStemmer


def execute():
    print('==================== Parte 1: Processamento de Email ====================')
    print('Preprocessando sample email (emailSample3.txt)...')

    with open('emails_teste/emailSample3.txt') as f:
        file_contents = f.read().rstrip('\n')

    word_indices = process_email(file_contents)

    print ('\n\n\n\n==================== Parte 2: Extracao de Feature ====================')
    print ('Extracting features from sample email (emailSample3.txt)...')
    features = email_features(word_indices)

    print ('Length of feature vector:', len(features))
    print('Number of non-zero entries:', np.sum(features > 0))

    print ('\n\n\n\n=========== Parte 3: Treinar SVM LInear para Spam Classification ========')
    # Load the Spam Email dataset
    mat_data = sio.loadmat('spamTrain.mat')
    x = mat_data['X']
    y = mat_data['y'].ravel()

    print ('Treinamento SVM Linear (Classificacao Spam)...')
    C = 0.1
    clf = svm.LinearSVC(C=C)
    clf.fit(x, y)
    p = clf.predict(x)
    print('Acuracia de Treinamento:', np.mean(p == y) * 100)

    print ('\n\n\n\n=================== Parte 4: Teste Spam Classification ================')
    # Load the test dataset
    mat_data = sio.loadmat('spamTest.mat')
    X_test = mat_data['Xtest']
    y_test = mat_data['ytest'].ravel()

    print ('Avaliando e treinando SVM linear com set de teste...')
    p = clf.predict(X_test)
    print ('Acuracia do teste:', np.mean(p == y_test) * 100)

    print ('\n\n\n\n================= Parte 5: Top Preditores de Spam ====================')
    coef = clf.coef_.ravel()
    idx = coef.argsort()[::-1]
    vocab_list = get_vocablist()

    print('Top preditores de Spam:')
    for i in range(15):
        print ("{0:} ({1:f})".format(vocab_list[idx[i]], coef[idx[i]]))

    print ('\n\n\n\n=================== Parte 6: Nossos emails =====================')
    filename = 'emails_teste/emailSample3.txt'
    with open(filename) as f:
        file_contents = f.read().rstrip('\n')

    word_indices = process_email(file_contents)
    x = email_features(word_indices)
    p = clf.predict(x.T)
    print('Processado:', filename, '\nClassificacao Spam:', p)
    print('(1 indica spam, 0 indica nao spam)')


def split(delimiters, string, maxsplit=0):
    pattern = '|'.join(map(re.escape, delimiters))
    return re.split(pattern, string, maxsplit)

#@Param email_contents:string  The email content
#@Return list A list of word indices.
def process_email(email_contents):

    vocab_list = get_vocablist()

    email_contents = email_contents.lower()
    email_contents = re.sub('<[^<>]+>', ' ', email_contents)
    email_contents = re.sub('[0-9]+', ' number ', email_contents)
    email_contents = re.sub('(http|https)://[^\s]*', ' httpaddr ', email_contents)
    email_contents = re.sub('[^\s]+@[^\s]+', 'emailaddr', email_contents)
    email_contents = re.sub('[$]+', ' dollar ', email_contents)

    words = split(""" @$/#.-:&*+=[]?!(){},'">_<;%\n\r""", email_contents)
    word_indices = []
    stemmer = PorterStemmer()
    print('Indices das palavras:')
    for word in words:
        word = re.sub('[^a-zA-Z0-9]', '', word)
        if word == '':
            continue
        word = stemmer.stem(word)
        if word in vocab_list:
            idx = vocab_list.index(word)
            word_indices.append(idx)
            print ( " (" , idx, ")" ,word , end = ', ')

    return word_indices

#@Param word_indices:array-like List of word indices.
#@Return  ndarray Feature vector from word indices.
def email_features(word_indices):
    # Total number of words in the dictionary
    n = 1899
    x = np.zeros((n, 1))
    x[word_indices] = 1

    return x

#@Return list The vocabulary list.
def get_vocablist():
    vocabulary = []
    with open('vocab.txt') as f:
        for line in f:
            idx, word = line.split('\t')
            vocabulary.append(word.strip())
    return vocabulary


execute()
