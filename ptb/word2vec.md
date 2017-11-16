
## ptb数据集 word2vec


```python
import collections
import math
import os
import random
import zipfile
import numpy as np
import urllib
import tensorflow as tf
```


```python
url = 'http://mattmahoney.net/dc/'

def maybe_download(filename,expected_bytes):
    if not os.path.exists(filename):
        filename,_ = urllib.request.urlretrieve(url + filename,filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified',filename)
    else:
        print(statinfo.st_size)
        raise Exception(\
            'Failed to verify '+ filename+'. Can you get to it with a browser?')
    return filename

filename = maybe_download('text8.zip',31344016)
    
```

    Found and verified text8.zip
    


```python
def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

words = read_data(filename)
print('Data size',len(words))
```

    Data size 17005207
    


```python
vocabulary_size = 50000

def bulid_dataset(words):
    count =[['UNK',-1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size -1))
    
    dictionary = dict()
    for word,_ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
            
        else:
            index =0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(),dictionary.keys()))
    
    return data,count,dictionary,reverse_dictionary

data,count,dictionary,reverse_dictionary = bulid_dataset(words)
```


```python
del words

print('Most common words(+UNK)',count[:8])
print('Sample data',data[:10],[reverse_dictionary[i] for i in data[:10]])
```

    Most common words(+UNK) [['UNK', 418391], ('the', 1061396), ('of', 593677), ('and', 416629), ('one', 411764), ('in', 372201), ('a', 325873), ('to', 316376)]
    Sample data [5241, 3081, 12, 6, 195, 2, 3135, 46, 59, 156] ['anarchism', 'originated', 'as', 'a', 'term', 'of', 'abuse', 'first', 'used', 'against']
    


```python
data_index = 0 # global 变量，单词序号

def generate_batch(batch_size, num_skips,skip_window):
    # batch_size 为 batch 大小
    # num_skips 每个单词生成多少个样本
    # skip_window 窗口大小
    
    global data_index
    
    assert batch_size % num_skips == 0
    assert num_skips <= 2*skip_window
    batch = np.ndarray(shape=(batch_size),dtype = np.int32)
    labels = np.ndarray(shape =(batch_size,1),dtype = np.int32)
    
    span = 2*skip_window +1 # 和某个单词关联的单词数量
    buffer = collections.deque(maxlen =span)
    
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index +1) %len(data)
    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]
        
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0,span -1)
            targets_to_avoid.append(target)
            batch[i*num_skips + j] = buffer[skip_window]
            labels[i*num_skips + j,0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index +1) % len(data)
    return batch, labels

batch, labels = generate_batch(batch_size = 8 ,num_skips = 4,skip_window = 2)

for i in range(8):
    print(batch[i],reverse_dictionary[batch[i]],'-->',labels[i,0],\
         reverse_dictionary[labels[i,0]])
```

    12 as --> 3081 originated
    12 as --> 195 term
    12 as --> 5241 anarchism
    12 as --> 6 a
    6 a --> 3081 originated
    6 a --> 2 of
    6 a --> 12 as
    6 a --> 195 term
    


```python
batch_size = 128
embedding_size = 128
skip_window = 1
num_skips = 2

valid_size = 16
valid_window = 100
valid_examples = np.random.choice(valid_window,valid_size,replace=False)
num_sampled = 64

```


```python
graph = tf.Graph()
with graph.as_default():
    
    train_inputs = tf.placeholder(tf.int32,shape=[batch_size])
    train_labels = tf.placeholder(tf.int32,shape=[batch_size,1])
    valid_dataset = tf.constant(valid_examples,dtype=tf.int32)
    
    with tf.device('/cpu:0'):
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size,embedding_size],-1.0,1.0))
    embed = tf.nn.embedding_lookup(embeddings,train_inputs)
    
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size,embedding_size],
                       stddev = 1.0/math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
    
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                    biases=nce_biases,
                                    labels=train_labels,
                                    inputs=embed,
                                    num_sampled=num_sampled,
                                    num_classes=vocabulary_size))
    
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings),1,keep_dims=True))

    normalized_embeddings = embeddings / norm

    valid_embeddings = tf.nn.embedding_lookup(
    normalized_embeddings,valid_dataset)
    similarity = tf.matmul(valid_embeddings,normalized_embeddings,transpose_b=True)

    init = tf.global_variables_initializer()

```


```python
num_steps = 100001

with tf.Session(graph=graph) as sess:
    sess.run(init)
    print("Initialized")
    
    average_loss = 0
    for step in range(num_steps):
        batch_inputs,batch_labels = generate_batch(
         batch_size,num_skips,skip_window)
        feed_dict = {train_inputs:batch_inputs,train_labels:batch_labels}
        
        _, loss_val = sess.run([optimizer,loss],feed_dict=feed_dict)
        average_loss += loss_val
        
        if step % 2000 ==0:
            if step >0:
                average_loss /=2000
            print("Average loss at step ",step,":",average_loss)
            average_loss = 0    
        
        if step % 10000 ==0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                
                top_k = 8
                nearest = (-sim[i,:]).argsort()[1:top_k +1]
                log_str = " Nearest to %s:"%valid_word
                
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = "%s %s,"%(log_str,close_word)
                print(log_str)
                
    final_embeddings = normalized_embeddings.eval()
    
```

    Initialized
    Average loss at step  0 : 271.991516113
     Nearest to were: meeting, boz, gana, father, soothe, glaciation, cycles, forecasting,
     Nearest to zero: plugboard, showing, dinneen, stiller, bigg, converting, primo, odie,
     Nearest to been: kg, hoyle, nanoscale, stockpiles, brahmins, packing, sentimental, submarines,
     Nearest to used: larry, progressions, voicing, disparities, unprovoked, stalks, cat, deianira,
     Nearest to most: toussaint, shorten, breatharian, duopoly, cramer, sequestration, pertwee, diachronic,
     Nearest to its: derby, fortifications, spc, cocos, organise, escoffier, slayer, cereals,
     Nearest to it: thibetanus, hellman, location, union, encircled, wickedness, novice, extensionality,
     Nearest to their: ibrd, tunic, sundarbans, catalog, treize, forseti, maelstrom, indochina,
     Nearest to four: carnegie, capybaras, defections, arras, initiatives, emden, overpower, choctaws,
     Nearest to these: hoyle, augusta, rectangular, immersive, alkaloid, infante, insuring, buick,
     Nearest to his: journalists, anand, krafft, lula, rock, artery, bowen, involuntarily,
     Nearest to as: richest, isotropic, expositor, boucher, oscar, bruton, homage, ken,
     Nearest to not: corridor, audiobook, contractors, hijacker, riddler, kew, ostrogoths, large,
     Nearest to and: inverses, storyline, mill, kanal, gram, stair, playboy, somali,
     Nearest to they: rearmament, wangenheim, legionnaire, tinymud, z, theorized, moc, seidel,
     Nearest to during: bernstein, lj, lafcadio, kapellmeister, moth, homophobic, variations, ruby,
    Average loss at step  2000 : 115.200463344
    Average loss at step  4000 : 52.5689281006
    Average loss at step  6000 : 33.0014310074
    Average loss at step  8000 : 23.5455425274
    Average loss at step  10000 : 16.9532447844
     Nearest to were: is, and, meeting, are, in, leftover, would, forecasting,
     Nearest to zero: nine, imdb, macabre, compression, iso, rho, three, foreground,
     Nearest to been: kg, olympian, merchant, predominates, rho, projector, dhimmis, submarines,
     Nearest to used: larry, discuss, rave, cat, decision, normalization, fitness, trick,
     Nearest to most: party, originated, means, dictionaries, serpent, imdb, displacement, toei,
     Nearest to its: derby, wells, netbios, the, pronounced, conversely, florida, residents,
     Nearest to it: which, they, union, boost, fc, differentiable, conflicts, known,
     Nearest to their: the, died, symphony, super, catalog, stratford, ibrd, phoenix,
     Nearest to four: two, diagrams, imdb, eight, seven, half, netbios, dragonball,
     Nearest to these: project, killer, excerpted, stated, spring, playing, coding, productive,
     Nearest to his: the, rock, antarctic, a, futures, flies, live, serving,
     Nearest to as: and, in, is, for, richest, netbios, by, revenge,
     Nearest to not: sousa, importance, anything, also, rho, boleslaus, large, corridor,
     Nearest to and: in, of, nine, the, from, for, UNK, as,
     Nearest to they: it, papers, being, he, tusks, not, antarctica, diaeresis,
     Nearest to during: bernstein, berman, chips, issue, cordilleras, cincinnati, variations, ruby,
    Average loss at step  12000 : 13.453854722
    Average loss at step  14000 : 11.7622537979
    Average loss at step  16000 : 10.3030331206
    Average loss at step  18000 : 8.76592329407
    Average loss at step  20000 : 8.00592013931
     Nearest to were: are, is, was, and, forecasting, leftover, would, by,
     Nearest to zero: nine, eight, five, six, seven, four, three, eml,
     Nearest to been: kg, crowning, merchant, olympian, who, was, voltages, rho,
     Nearest to used: discuss, larry, cat, trick, rave, tansley, parl, torture,
     Nearest to most: originated, party, cad, wikisource, displacement, serpent, paved, world,
     Nearest to its: the, their, ecclesia, conversely, wells, derby, ebu, smells,
     Nearest to it: which, this, he, they, she, boost, erlk, exerts,
     Nearest to their: the, his, died, its, essenes, sake, catalog, phoenix,
     Nearest to four: eight, two, seven, nine, zero, five, six, one,
     Nearest to these: killer, alliances, excerpted, all, project, catalonia, spring, haven,
     Nearest to his: the, their, rock, depends, flies, a, one, antarctic,
     Nearest to as: is, by, and, in, for, netbios, sld, leith,
     Nearest to not: also, sousa, there, they, to, golgi, importance, contractors,
     Nearest to and: or, in, of, eml, UNK, for, s, with,
     Nearest to they: it, he, not, there, papers, that, who, tusks,
     Nearest to during: bernstein, of, berman, and, wanna, disestablishment, chips, tyranny,
    Average loss at step  22000 : 7.47506773686
    Average loss at step  24000 : 6.80932211971
    Average loss at step  26000 : 6.40562643671
    Average loss at step  28000 : 6.29945880783
    Average loss at step  30000 : 6.21676191926
     Nearest to were: are, was, is, and, forecasting, by, homosexuals, extremophiles,
     Nearest to zero: nine, eight, five, six, four, seven, three, quagga,
     Nearest to been: was, be, kg, by, were, hemionus, had, who,
     Nearest to used: neutronic, maja, discuss, unprovoked, tansley, larry, cheese, fitness,
     Nearest to most: ass, bfbs, originated, cad, cooh, paved, wikisource, faber,
     Nearest to its: the, their, his, conversely, wells, smells, ecclesia, nuclear,
     Nearest to it: he, this, which, they, she, there, ass, not,
     Nearest to their: his, the, its, died, essenes, hemionus, sake, a,
     Nearest to four: two, eight, five, six, seven, three, zero, nine,
     Nearest to these: all, excerpted, some, the, alliances, killer, hellenistic, catalonia,
     Nearest to his: the, their, her, its, depends, neutronic, flies, a,
     Nearest to as: is, for, and, by, in, neutronic, netbios, memnon,
     Nearest to not: also, there, it, they, to, sousa, golgi, importance,
     Nearest to and: or, in, ass, eml, neutronic, quagga, maja, s,
     Nearest to they: he, it, there, who, not, that, which, banshee,
     Nearest to during: bernstein, in, of, berman, from, disestablishment, wanna, at,
    Average loss at step  32000 : 5.85848354912
    Average loss at step  34000 : 5.62263574159
    Average loss at step  36000 : 6.11974466252
    Average loss at step  38000 : 5.535706357
    Average loss at step  40000 : 5.45226627553
     Nearest to were: are, was, is, have, homosexuals, had, patience, be,
     Nearest to zero: eight, five, seven, nine, six, four, three, two,
     Nearest to been: be, was, were, by, had, gts, hemionus, also,
     Nearest to used: unprovoked, cip, neutronic, discuss, tansley, fitness, maja, headmaster,
     Nearest to most: ass, more, bfbs, paved, originated, wikisource, cooh, augment,
     Nearest to its: their, the, his, arctocephalus, conversely, smells, reykjav, ecclesia,
     Nearest to it: he, this, which, there, she, they, ass, arctocephalus,
     Nearest to their: his, the, its, died, essenes, arctocephalus, hemionus, stratford,
     Nearest to four: seven, three, six, eight, five, two, zero, one,
     Nearest to these: all, some, yourdon, many, alliances, excerpted, the, two,
     Nearest to his: their, the, her, its, depends, arctocephalus, s, smells,
     Nearest to as: for, is, netbios, neutronic, by, and, krzysztof, with,
     Nearest to not: also, it, they, there, barb, to, golgi, sousa,
     Nearest to and: or, but, neutronic, in, ass, quagga, maja, arctocephalus,
     Nearest to they: he, there, it, who, you, not, banshee, that,
     Nearest to during: in, bernstein, berman, at, from, of, ruby, disestablishment,
    Average loss at step  42000 : 5.30276678836
    Average loss at step  44000 : 5.04017378008
    Average loss at step  46000 : 5.2136328994
    Average loss at step  48000 : 5.35128249753
    Average loss at step  50000 : 5.23867391741
     Nearest to were: are, was, have, is, had, homosexuals, by, be,
     Nearest to zero: eight, five, six, seven, nine, four, three, stadtbahn,
     Nearest to been: be, was, by, were, had, valved, gts, also,
     Nearest to used: unprovoked, neutronic, tansley, called, cip, stadtbahn, considered, zilog,
     Nearest to most: ass, more, bfbs, wikisource, paved, cooh, originated, wedding,
     Nearest to its: their, the, his, arctocephalus, smells, conversely, reykjav, her,
     Nearest to it: he, this, there, which, she, they, ass, arctocephalus,
     Nearest to their: its, his, the, essenes, arctocephalus, died, modula, heim,
     Nearest to four: five, six, seven, three, eight, two, nine, zero,
     Nearest to these: some, all, many, yourdon, two, three, excerpted, such,
     Nearest to his: their, the, its, her, depends, smells, s, arctocephalus,
     Nearest to as: sld, netbios, neutronic, soi, kiang, yourdon, is, for,
     Nearest to not: also, they, it, to, there, barb, golgi, generally,
     Nearest to and: or, but, in, neutronic, quagga, ass, six, arctocephalus,
     Nearest to they: he, there, it, who, you, not, banshee, catastrophic,
     Nearest to during: in, bernstein, from, at, after, of, berman, and,
    Average loss at step  52000 : 5.24991915441
    Average loss at step  54000 : 5.07710778058
    Average loss at step  56000 : 5.08117939663
    Average loss at step  58000 : 4.83502101886
    Average loss at step  60000 : 4.92004659224
     Nearest to were: are, was, have, had, is, by, be, been,
     Nearest to zero: seven, six, eight, five, nine, four, three, stadtbahn,
     Nearest to been: be, was, were, had, valved, by, motorsports, also,
     Nearest to used: diaphragm, unprovoked, called, considered, tansley, known, neutronic, presenter,
     Nearest to most: more, ass, bfbs, wedding, wikisource, hemionus, paved, augment,
     Nearest to its: their, the, his, her, reykjav, arctocephalus, smells, abelard,
     Nearest to it: he, this, there, she, which, they, ass, not,
     Nearest to their: its, his, the, her, essenes, some, sake, arctocephalus,
     Nearest to four: five, seven, six, three, eight, two, one, nine,
     Nearest to these: some, all, many, such, yourdon, two, which, several,
     Nearest to his: their, her, the, its, depends, s, smells, arctocephalus,
     Nearest to as: when, netbios, sld, memnon, soi, hemionus, neutronic, kiang,
     Nearest to not: they, it, also, generally, barb, there, golgi, you,
     Nearest to and: or, but, quagga, neutronic, arctocephalus, ass, eml, hemionus,
     Nearest to they: he, there, it, who, you, not, banshee, these,
     Nearest to during: in, after, from, at, on, with, bernstein, berman,
    Average loss at step  62000 : 5.02678824782
    Average loss at step  64000 : 4.87538512218
    Average loss at step  66000 : 4.91488063014
    Average loss at step  68000 : 4.91848220146
    Average loss at step  70000 : 4.84098144293
     Nearest to were: are, was, have, had, be, by, believe, been,
     Nearest to zero: five, eight, seven, six, four, nine, three, landesverband,
     Nearest to been: be, was, were, had, by, valved, motorsports, nine,
     Nearest to used: diaphragm, unprovoked, considered, known, called, hyi, tansley, presenter,
     Nearest to most: more, ass, bfbs, many, wedding, hemionus, augment, wikisource,
     Nearest to its: their, the, his, her, landesverband, hyi, reykjav, landsmannschaft,
     Nearest to it: he, this, there, she, which, ass, they, landesverband,
     Nearest to their: its, his, the, her, some, landesverband, hyi, sake,
     Nearest to four: five, six, three, seven, eight, zero, nine, two,
     Nearest to these: some, many, all, such, several, yourdon, they, there,
     Nearest to his: their, her, its, the, depends, s, flies, smells,
     Nearest to as: when, landesverband, by, in, neutronic, hemionus, sld, netbios,
     Nearest to not: they, also, barb, it, generally, you, golgi, there,
     Nearest to and: or, but, landesverband, in, eml, ass, neutronic, quagga,
     Nearest to they: he, there, you, who, it, not, we, she,
     Nearest to during: in, after, at, from, on, through, when, landesverband,
    Average loss at step  72000 : 4.88546256185
    Average loss at step  74000 : 4.82114040458
    Average loss at step  76000 : 4.79165117002
    Average loss at step  78000 : 4.6168329711
    Average loss at step  80000 : 4.7330746721
     Nearest to were: are, was, have, had, believe, been, be, by,
     Nearest to zero: six, five, seven, four, eight, nine, landesverband, three,
     Nearest to been: be, was, were, had, valved, by, motorsports, gts,
     Nearest to used: known, called, unprovoked, considered, diaphragm, tansley, hyi, presenter,
     Nearest to most: more, ass, many, bfbs, some, wedding, augment, hemionus,
     Nearest to its: their, the, his, her, landesverband, reykjav, hyi, smells,
     Nearest to it: he, this, there, she, which, ils, they, ass,
     Nearest to their: its, his, the, her, some, sake, landesverband, arctocephalus,
     Nearest to four: five, six, three, seven, eight, zero, two, nine,
     Nearest to these: some, many, all, such, several, yourdon, they, both,
     Nearest to his: their, her, its, the, s, depends, my, flies,
     Nearest to as: by, landesverband, neutronic, when, hemionus, netbios, quagga, without,
     Nearest to not: they, also, it, generally, barb, to, you, parl,
     Nearest to and: or, landesverband, but, jumaada, neutronic, quagga, hyi, arctocephalus,
     Nearest to they: he, there, you, who, it, not, we, she,
     Nearest to during: in, after, at, from, through, when, on, neutronic,
    Average loss at step  82000 : 4.73962371957
    Average loss at step  84000 : 4.73558593023
    Average loss at step  86000 : 4.74403536713
    Average loss at step  88000 : 4.65004122233
    Average loss at step  90000 : 4.61863064349
     Nearest to were: are, was, have, had, be, is, been, seven,
     Nearest to zero: seven, six, five, nine, landesverband, four, eight, stadtbahn,
     Nearest to been: be, was, had, were, valved, become, by, nine,
     Nearest to used: known, called, considered, tansley, unprovoked, presenter, hyi, diaphragm,
     Nearest to most: more, many, ass, bfbs, some, pesce, wedding, augment,
     Nearest to its: their, his, the, her, rsc, reykjav, landesverband, our,
     Nearest to it: he, this, there, she, which, they, ils, ass,
     Nearest to their: its, his, her, the, pesce, some, landesverband, arctocephalus,
     Nearest to four: five, three, six, seven, eight, two, landesverband, nine,
     Nearest to these: some, many, all, such, several, both, there, yourdon,
     Nearest to his: their, her, its, the, s, smells, depends, my,
     Nearest to as: landesverband, neutronic, netbios, sld, by, quagga, in, hemionus,
     Nearest to not: they, generally, often, also, it, you, typically, peculiar,
     Nearest to and: or, landesverband, but, quagga, one, hemionus, six, jumaada,
     Nearest to they: he, there, you, it, who, we, not, she,
     Nearest to during: in, after, at, when, from, through, on, neutronic,
    Average loss at step  92000 : 4.2824965831
    Average loss at step  94000 : 4.50002410281
    Average loss at step  96000 : 4.43510123587
    Average loss at step  98000 : 4.53519476604
    Average loss at step  100000 : 4.55222624135
     Nearest to were: are, was, have, had, be, believe, is, been,
     Nearest to zero: six, five, seven, eight, four, nine, three, landesverband,
     Nearest to been: be, was, had, were, become, valved, by, federalists,
     Nearest to used: known, called, hyi, tansley, considered, diaphragm, unprovoked, presenter,
     Nearest to most: more, some, ass, many, bfbs, pesce, augment, use,
     Nearest to its: their, his, the, her, rsc, reykjav, our, landesverband,
     Nearest to it: he, this, she, there, which, they, ass, ils,
     Nearest to their: its, his, the, her, some, your, pesce, these,
     Nearest to four: five, six, seven, three, eight, two, nine, zero,
     Nearest to these: some, many, all, such, both, several, the, their,
     Nearest to his: their, her, its, the, s, smells, landesverband, my,
     Nearest to as: netbios, without, landesverband, neutronic, by, anh, when, sld,
     Nearest to not: they, it, you, to, generally, also, often, be,
     Nearest to and: but, or, landesverband, arctocephalus, quagga, ass, jumaada, which,
     Nearest to they: he, there, you, who, it, we, she, not,
     Nearest to during: after, in, at, when, through, from, neutronic, on,
    


```python
def plot_with_labels(low_dim_embs,labels,filename='tsne.png'):
    assert low_dim_embs.shape[0] >=len(labels),"More labels than embeddings"
    plt.figure(figsize =(18,18))
    for i, label in enumerate(labels):
        x,y = low_dim_embs[i,:]
        plt.scatter(x,y)
        plt.annotate(label,xy=(x,y),xytext =(5,2),textcoords = 'offset points',
                    ha ='right',va = 'bottom')
        plt.savefig(filename)
    
```


```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE(perplexity=30,n_components=2,init='pca',n_iter=5000)

plot_only = 300
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
labels = [reverse_dictionary[i] for i in range(plot_only)]
plot_with_labels(low_dim_embs,labels)
```
