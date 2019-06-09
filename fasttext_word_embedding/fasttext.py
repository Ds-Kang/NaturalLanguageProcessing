import torch
import random
from collections import Counter
import argparse
from math import sqrt

def make_ngram(word,n):
    ngram=set()
    for i in range(len(word)-n+1):
        if i<len(word)-n:
            ngram.add(word[i:i+n])
        else:
            ngram.add(word[i:i+n-1]+' ')
        ngram.add(' '+word[0:n-1])
    return ngram

def make_subword(word):
    gram3=set(make_ngram(word,3))
    gram4=set(make_ngram(word,4))
    gram5=set(make_ngram(word,5))
    gram6=set(make_ngram(word,6))
    special=set([word])

    grams=[gram3,gram4,gram5,gram6,special]

    return grams

def FNV1a_hashing(word):
    FNV_prime=16777619
    hashe = 2166136261 #내장함수와 구별위해
    for char in word:
        hashe = hashe ^ ord(char)
        hashe = hashe * FNV_prime
    return hashe


def embedding_for_index(emb,vocab):
    #word to hash
    w2h = {}
    for word in vocab:
        hashes=[]
        grams=make_subword(word)
        for gram in grams:
            for subword in gram:
                hashes.append(FNV1a_hashing(subword)%2100000)
        w2h[word]=hashes


    emb_i = torch.zeros(len(vocab), 300)
    i=0
    for word in vocab:
        emb_i[i]=torch.sum(emb[w2h[word]],0)
        i+=1

    return emb_i


def find_similar_word(embedding,emb_i,w2i,i2w,part,ns):
    question_words = open('questions-words.txt',mode='r').read().split('\n')
    result =open('result_{}_{}.txt'.format(ns,part),mode='w')

    print(question_words)
    for word in question_words:
        if(word==''):
            continue

        word_emb=torch.zeros_like(emb_i[0])
        grams=make_subword(word)
        for gram in grams:
            for subword in gram:
                word_emb+=embedding[FNV1a_hashing(subword)%2100000]

        x=word_emb

        length = (emb_i*emb_i).sum(1)**0.5 #cosine similarity 계산 위한 크기
        inputVector = torch.unsqueeze(x,0)/(x*x).sum()**0.5
        sim = (inputVector@emb_i.t())[0]/length #cosine similarity 계산
        values, indices = sim.squeeze().topk(5) #상위 5개 저장


        result.write("\n")
        result.write("===============================================\n")
        result.write("The most similar words to \"" + word + "\"\n")
        for ind, val in zip(indices,values):
            result.write(i2w[ind.item()]+": %.3f"%(val,)+"\n")
        result.write("===============================================\n")
        result.write("\n")

    result.close()
    
def subsampling(input_seq,target_seq):
###############################  Output  #########################################
# subsampled : Subsampled sequence                                               #
##################################################################################

    subsampled_input_seq=[]
    subsampled_target_seq=[]
    t=1e-5 #subsamping 상수 t

    window_size=0 #SG에서는 input seq가 window size*2만큼씩 중복되기 때문에 window size를 측정
    for target in target_seq:
        if target==0:
            window_size+=1
        else:
            break

    freq=Counter(input_seq) #frequent table 만들기 위해 단어 개수 측정
    cnt=0
    for inputs, target in zip(input_seq,target_seq):
        if cnt%(window_size*2)==0: #input seq의 중복 단어 중 첫 한개의 단어에 대해서만 확률 계산
            p=0
            if (1-sqrt(t/(freq[inputs]/window_size)))>random.random(): #확률에 따른 sequence 값 변경
                subsampled_input_seq.append(inputs)
                subsampled_target_seq.append(target)
                p=1
        else:
            if p==1:
                subsampled_input_seq.append(inputs)
                subsampled_target_seq.append(target)

    return subsampled_input_seq, subsampled_target_seq

def skipgram_NS(centerWord, inputMatrix, outputMatrix):
################################  Input  ##########################################
# centerWord : Index of a centerword (type:int)                                   #
# inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))                   #
# outputMatrix : Activated weight matrix of output (type:torch.tesnor(K,D))       #
###################################################################################

###############################  Output  ##########################################
# loss : Loss value (type:torch.tensor(1))                                        #
# grad_in : Gradient of inputMatrix (type:torch.tensor(1,D))                      #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(K,D))                    #
###################################################################################
 
    score=torch.mm(torch.sum(inputMatrix,0,keepdim=True),outputMatrix.t())
    sigmoid = 1/(1+torch.exp(score)) #오답인 경우 sigmoid 함수에 -x값을 대입
    sigmoid[0][0] = 1-sigmoid[0][0] #정답인 경우 올바른 sigmoid 함수값 나타나도록 변경

    loss=torch.sum(-torch.log(sigmoid))
    grad=torch.zeros_like(sigmoid)
    grad[0][0]=(sigmoid[0][0]-1) #정답인 경우의 gradient
    grad[0][1:]=(1-sigmoid[0][1:]) #오답인 경우의 gradient

    grad_in=torch.mm(grad,outputMatrix)
    grad_out=torch.mm(grad.t(),torch.sum(inputMatrix,0,keepdim=True))

    return loss, grad_in, grad_out


def word2vec_trainer(input_seq, target_seq, numwords, stats, i2w, NS=20, dimension=100, learning_rate=0.025, epoch=3):
# train_seq : list(tuple(int, list(int))

# Xavier initialization of weight matrices
    W_in = torch.randn(2100000, dimension) / (dimension**0.5)
    W_out = torch.randn(numwords, dimension) / (dimension**0.5)
    i=0
    losses=[]
    print("# of training samples")
    print(len(input_seq))
    print()
    stats = torch.LongTensor(stats)

    for _ in range(epoch):
        #subsampling
        input_seq,target_seq =subsampling(input_seq,target_seq)

        #Training word2vec using SGD(Batch size : 1)
        for inputs, output in zip(input_seq,target_seq):
            i+=1
            #Only use the activated rows of the weight matrix
            #activated should be torch.tensor(K,) so that activated W_out has the form of torch.tensor(K, D)
            words=[output] #output의 첫 단어를 맞추는 단어로 나머지를 오답인 단어로 설정
            for _ in range(NS):
                while 1:    
                    word = random.choice(stats) #frequent table에서 임의로 단어 추출
                    if word not in words: #중복 단어 추출되지 않도록 설정
                        words.append(word)
                        break
            activated=torch.tensor(words)

            hashes=[]
            grams=make_subword(i2w[inputs])
            for gram in grams:
                for subword in gram:
                    hashes.append(FNV1a_hashing(subword)%2100000)
            #ngram of W_in
            hashes=torch.tensor(hashes)
            L, G_in, G_out = skipgram_NS(inputs, W_in[hashes], W_out[activated])
            W_in[hashes] -= learning_rate*G_in
            W_out[activated] -= learning_rate*G_out

            
            losses.append(L.item())
            if i%50000==0:
                avg_loss=sum(losses)/len(losses)
                print("Loss : %f" %(avg_loss,))
                print("Percent:",i/len(input_seq))
                losses=[]

    return W_in, W_out


def main():
    parser = argparse.ArgumentParser(description='Word2vec')
    parser.add_argument('ns', metavar='negative_samples', type=int,
                        help='0 for hierarchical softmax, the other numbers would be the number of negative samples')
    parser.add_argument('part', metavar='partition', type=str,
                        help='"part" if you want to train on a part of corpus, "full" if you want to train on full corpus')
    args = parser.parse_args()
    ns= args.ns
    part = args.part

    #Load and preprocess corpus
    print("loading...")
    if part=="part":
        text = open('text8',mode='r').readlines()[0][:1000000] #Load a part of corpus for debugging
    elif part=="full":
        text = open('text8',mode='r').readlines()[0] #Load full corpus for submission
    else:
        print("Unknown argument : " + part)
        exit()

    print("preprocessing...")

    corpus = text.split()
    stats = Counter(corpus)
    words = []

    #Discard rare words
    for word in corpus:
        if stats[word]>4:
            words.append(word)
    vocab = set(words)

    #Give an index number to a word
    w2i = {}
    w2i[" "]=0
    i = 1
    for word in vocab:
        w2i[word] = i
        i+=1
    i2w = {}
    for k,v in w2i.items():
        i2w[v]=k

    

    #Frequency table for negative sampling
    freqtable = [0,0,0]
    for k,v in stats.items():
        f = int(v**0.75)
        for _ in range(f):
            if k in w2i.keys():
                freqtable.append(w2i[k])

    #Make training set
    print("build training set...")
    input_set = []
    target_set = []
    window_size = 2
    for j in range(len(words)):
        if j<window_size:
            input_set += [w2i[words[j]] for _ in range(window_size*2)]
            target_set += [0 for _ in range(window_size-j)] + [w2i[words[k]] for k in range(j)] + [w2i[words[j+k+1]] for k in range(window_size)]
        elif j>=len(words)-window_size:
            input_set += [w2i[words[j]] for _ in range(window_size*2)]
            target_set += [w2i[words[j-k-1]] for k in range(window_size)] + [w2i[words[len(words)-k-1]] for k in range(len(words)-j-1)] + [0 for _ in range(j+window_size-len(words)+1)]
        else:
            input_set += [w2i[words[j]] for _ in range(window_size*2)]
            target_set += [w2i[words[j-k-1]] for k in range(window_size)] + [w2i[words[j+k+1]] for k in range(window_size)]

    print("Vocabulary size")
    print(len(w2i))
    print()

    #Training section
    emb,_ = word2vec_trainer(input_set, target_set, len(w2i), freqtable, i2w, NS=ns, dimension=300, epoch=1, learning_rate=0.01)
    emb_i=embedding_for_index(emb,vocab)

    find_similar_word(emb,emb_i,w2i,i2w,part,ns)


main()