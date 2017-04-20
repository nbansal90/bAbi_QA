
# coding: utf-8

# author @Nitin Bansal


##Preprocessing Part has been removed since it remains the same


#Selecting the first file of Both Train and test
for i in range(0,20):
    tra = files_train[i]
    tes = files_test[i]

    os.chdir(path)

    train = open(tra, "r")
    test = open(tes,"r")

    #Getting all the words present in the Train and Test file respectively
    train = get_stories(train,True)
    test = get_stories(test)


    vocab = set()
    for story, q, answer in train + test:
        vocab |= set(story + q + [answer])

    print ("Weights of the Embedding Layer")
    a1 = model.get_weights()
    print (type(a1))
    for i in range(0, len(a1)):
        print (np.array(a1[i]).shape)
        print (a1[i])

    print (len(a1))




sentence_embedding = a1[0]
question_embedding = a1[1]
c1 = np.concatenate((a1[0],a1[1]), axis=0)


# In[10]:

reverse_dictionary = {}
vocab1 = ['.', '?', 'Bill', 'Fred', 'Is', 'Julie', 'Mary', 'back', 'bedroom', 'cinema', 'either', 'in', 'is', 'journeyed', 'kitchen', 'maybe', 'moved', 'no', 'office', 'or', 'park', 'school', 'the', 'to', 'travelled', 'went', 'yes']
for i in range(0,len(vocab1)):
    reverse_dictionary[i] = vocab1[i]


# In[11]:

import matplotlib.pyplot as plt
def plot_with_labels(low_dim_embs, labels, filename='tsne_q.png'):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

    plt.savefig(filename)


# In[27]:

from sklearn.manifold import TSNE
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
plot_only = 27
low_dim_embs = tsne.fit_transform(c1[:plot_only, :])
labels = [reverse_dictionary[i] for i in xrange(plot_only)]
plot_with_labels(low_dim_embs, labels)

