import torch
import random
from collections import Counter
import argparse
from huffman import HuffmanCoding
from math import sqrt

def Analogical_Reasoning_Task(embedding,w2i,i2w,mode,part,ns):
#######################  Input  #########################
# embedding : Word embedding (type:torch.tesnor(V,D))   #
#########################################################
    question_word = open('questions-words.txt',mode='r').readlines()
    result =open('result_{}_{}_{}.txt'.format(mode,ns,part),mode='w')

    for question in question_word[1:]:
        words=question.split()
        OOV=0
        for word in words:
            if word not in w2i.keys():
                OOV=1
                break
        if(OOV==1):
            continue
        x=embedding[w2i[words[1]]]-embedding[w2i[words[0]]]+embedding[w2i[words[2]]]


        length = (embedding*embedding).sum(1)**0.5
        inputVector = torch.unsqueeze(x,0)/(x*x).sum()**0.5
        sim = (inputVector@embedding.t())[0]/length
        values, indices = sim.squeeze().topk(5)


        result.write("\n")
        result.write("===============================================\n")
        result.write("The most similar words to \"" + words[1] + "-" + words[0] + "+" + words[2]+ "\"\n")
        result.write("Answer is: "+ words[3]+"\n")
        for ind, val in zip(indices,values):
            result.write(i2w[ind.item()]+":%.3f"%(val,)+"\n")
        result.write("===============================================\n")
        result.write("\n")

    result.close()



def subsampling(text):
################################  Input  #########################################
# text : Original text (type:str)                                                #
##################################################################################

###############################  Output  #########################################
# subsampled : Subsampled text (type:list(str))                                  #
##################################################################################
    stats=text.split()
    freq=Counter(text)
    t=1e-5

    subsampled=[]
    for word in stats:
        if (1-sqrt(t/(freq[word]+1)))>random.random():
            subsampled.append(word)

    return subsampled

                                                            
def skipgram_HS(centerWord, contextCode, inputMatrix, outputMatrix):
################################  Input  ##########################################
# centerWord : Index of a centerword (type:int)                                   #
# contextCode : Code of a contextword (type:str)                                  #
# inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))                   #
# outputMatrix : Activated weight matrix of output (type:torch.tesnor(K,D))       #
###################################################################################

###############################  Output  ##########################################
# loss : Loss value (type:torch.tensor(1))                                        #
# grad_in : Gradient of inputMatrix (type:torch.tensor(1,D))                      #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(K,D))                    #
###################################################################################

    grad_in = torch.zeros(1,len(inputMatrix[centerWord]))
    grad_out = torch.zeros_like(outputMatrix)
    loss=0

    way=0
    for j in contextCode:
        if(int(j)==0):
            sig=1/(1+torch.exp(-torch.dot(inputMatrix[centerWord],outputMatrix[way])))
            grad_out[way]=(sig-1) * inputMatrix[centerWord]
            grad_in+=(sig-1) * outputMatrix[way]
            loss-=torch.log(sig)
        else:
            sig=1/(1+torch.exp(torch.dot(inputMatrix[centerWord],outputMatrix[way])))
            grad_out[way]=(1-sig) * inputMatrix[centerWord]
            grad_in+=(1-sig) * outputMatrix[way]
            loss-=torch.log(sig)
        way=way+1


    return loss, grad_in, grad_out



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
 
    score=torch.mm(torch.unsqueeze(inputMatrix[centerWord],0),outputMatrix.t())
    sigmoid = 1/(1+torch.exp(score))
    sigmoid[0][0] = 1-sigmoid[0][0]

    loss=torch.sum(-torch.log(sigmoid))
    grad=torch.zeros_like(sigmoid)
    grad[0][0]=(sigmoid[0][0]-1)
    grad[0][1:]=(1-sigmoid[0][1:])

    grad_in=torch.mm(grad,outputMatrix).t()
    grad_out=torch.mm(grad.t(),torch.unsqueeze(inputMatrix[centerWord],0))

    return loss, grad_in, grad_out


def CBOW_HS(contextWords, centerCode, inputMatrix, outputMatrix):
################################  Input  ##########################################
# contextWords : Indices of contextwords (type:list(int))                          #
# centerCode : Code of a centerword (type:str)                                    #
# inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))                   #
# outputMatrix : Activated Weight matrix of output (type:torch.tesnor(K,D))       #
###################################################################################

###############################  Output  ##########################################
# loss : Loss value (type:torch.tensor(1))                                        #
# grad_in : Gradient of inputMatrix (type:torch.tensor(1,D))                      #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(K,D))                    #
###################################################################################

    grad_in = torch.zeros(1,len(inputMatrix[contextWords[0]]))
    grad_out = torch.zeros_like(outputMatrix)
    loss=0

    way=0
    for j in centerCode:
        if(int(j)==0):
            sig=1/(1+torch.exp(-torch.dot(torch.mean(inputMatrix[contextWords],0),outputMatrix[way])))
            grad_out[way]=(sig-1) * torch.mean(inputMatrix[contextWords],0,keepdim=True)
            grad_in+=(sig-1) * outputMatrix[way]
            loss-=torch.log(sig)
        else:
            sig=1/(1+torch.exp(torch.dot(torch.mean(inputMatrix[contextWords],0),outputMatrix[way])))
            grad_out[way]=(1-sig) * torch.mean(inputMatrix[contextWords],0,keepdim=True)
            grad_in+=(1-sig) * outputMatrix[way]
            loss-=torch.log(sig)
        way=way+1

    return loss, grad_in, grad_out


def CBOW_NS(contextWords, inputMatrix, outputMatrix):
################################  Input  ##########################################
# contextWords : Indices of contextwords (type:list(int))                          #
# inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))                   #
# outputMatrix : Activated Weight matrix of output (type:torch.tesnor(K,D))       #
###################################################################################

###############################  Output  ##########################################
# loss : Loss value (type:torch.tensor(1))                                        #
# grad_in : Gradient of inputMatrix (type:torch.tensor(1,D))                      #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(K,D))                    #
###################################################################################

    score=torch.mm(torch.mean(inputMatrix[contextWords],0,keepdim=True),outputMatrix.t())
    sigmoid = 1/(1+torch.exp(score))
    sigmoid[0][0] = 1-sigmoid[0][0]

    loss=torch.sum(-torch.log(sigmoid))
    grad=torch.zeros_like(sigmoid)
    grad[0][0]=(sigmoid[0][0]-1)
    grad[0][1:]=(1-sigmoid[0][1:])

    grad_in=torch.mm(grad,outputMatrix)
    grad_out=torch.mm(grad.t(),torch.mean(inputMatrix[contextWords],0,keepdim=True))

    return loss, grad_in, grad_out


def word2vec_trainer(input_seq, target_seq, numwords, codes, stats, mode="CBOW", NS=20, dimension=100, learning_rate=0.025, epoch=3):
# train_seq : list(tuple(int, list(int))

# Xavier initialization of weight matrices
    W_in = torch.randn(numwords, dimension) / (dimension**0.5)
    if NS==0:
        W_out = torch.randn(2*numwords, dimension) / (dimension**0.5)
        ways=set()
        for code in codes.values():
            way=0
            for j in code:
                if int(j)==0:
                    way=2*way+1
                else:
                    way=2*way+2
                ways.add(way)
        way2i = {}
        i = 1
        for way in ways:
            way2i[way] = i
            i+=1

    else:
        W_out = torch.randn(numwords, dimension) / (dimension**0.5)
    i=0
    losses=[]
    print("# of training samples")
    print(len(input_seq))
    print()
    stats = torch.LongTensor(stats)

    for _ in range(epoch):
        #Training word2vec using SGD(Batch size : 1)
        for inputs, output in zip(input_seq,target_seq):
            i+=1
            if mode=="CBOW":
                if NS==0:
                    #Only use the activated rows of the weight matrix
                    #activated should be torch.tensor(K,) so that activated W_out has the form of torch.tensor(K, D)
                    ways = []
                    way=0
                    if output==0:
                        continue
                    for j in codes[output]:
                        if int(j)==0:
                            way=2*way+1
                        else:
                            way=2*way+2
                        ways.append(way2i[way])
                    activated=torch.tensor(ways)
                    L, G_in, G_out = CBOW_HS(inputs, codes[output], W_in, W_out[activated])
                    W_in[inputs] -= learning_rate*G_in
                    W_out[activated] -= learning_rate*G_out
                else:
                    #Only use the activated rows of the weight matrix
                    #activated should be torch.tensor(K,) so that activated W_out has the form of torch.tensor(K, D)
                    words=[output]
                    for _ in range(NS):
                        while 1:    
                            word = random.choice(stats)
                            if word not in words:
                                words.append(word)
                                break
                    activated=torch.tensor(words)
                    L, G_in, G_out = CBOW_NS(inputs, W_in, W_out[activated])
                    W_in[inputs] -= learning_rate*G_in
                    W_out[activated] -= learning_rate*G_out

            elif mode=="SG":
                if NS==0:
                    #Only use the activated rows of the weight matrix
                    #activated should be torch.tensor(K,) so that activated W_out has the form of torch.tensor(K, D)
                    ways = []
                    way=0
                    if output==0:
                        continue
                    for j in codes[output]:
                        if int(j)==0:
                            way=2*way+1
                        else:
                            way=2*way+2
                        ways.append(way2i[way])
                    activated=torch.tensor(ways)
                    L, G_in, G_out = skipgram_HS(inputs, codes[output], W_in, W_out[activated])
                    W_in[inputs] -= learning_rate*G_in.squeeze()
                    W_out[activated] -= learning_rate*G_out
                else:
                    #Only use the activated rows of the weight matrix
                    #activated should be torch.tensor(K,) so that activated W_out has the form of torch.tensor(K, D)
                    words=[output]
                    for _ in range(NS):
                        while 1:    
                            word = random.choice(stats)
                            if word not in words:
                                words.append(word)
                                break
                    activated=torch.tensor(words)
                    L, G_in, G_out = skipgram_NS(inputs, W_in, W_out[activated])
                    W_in[inputs] -= learning_rate*G_in.squeeze()
                    W_out[activated] -= learning_rate*G_out

                
            else:
                print("Unkwnown mode : "+mode)
                exit()
            losses.append(L.item())
            if i%50000==0:
                avg_loss=sum(losses)/len(losses)
                print("Loss : %f" %(avg_loss,))
                print("Percent:",i/len(input_seq))
                losses=[]

    return W_in, W_out


def main():
    parser = argparse.ArgumentParser(description='Word2vec')
    parser.add_argument('mode', metavar='mode', type=str,
                        help='"SG" for skipgram, "CBOW" for CBOW')
    parser.add_argument('ns', metavar='negative_samples', type=int,
                        help='0 for hierarchical softmax, the other numbers would be the number of negative samples')
    parser.add_argument('part', metavar='partition', type=str,
                        help='"part" if you want to train on a part of corpus, "full" if you want to train on full corpus')
    args = parser.parse_args()
    mode = args.mode
    part = args.part
    ns = args.ns

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

    #subsampling of frequent words
    corpus = subsampling(text)
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


    #Code dict for hierarchical softmax
    freqdict={}
    for word in vocab:
        freqdict[w2i[word]]=stats[word]
    codedict= HuffmanCoding().build(freqdict)

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
    window_size = 5
    if mode=="CBOW":
        for j in range(len(words)):
            if j<window_size:
                input_set.append([0 for _ in range(window_size-j)] + [w2i[words[k]] for k in range(j)] + [w2i[words[j+k+1]] for k in range(window_size)])
                target_set.append(w2i[words[j]])
            elif j>=len(words)-window_size:
                input_set.append([w2i[words[j-k-1]] for k in range(window_size)] + [w2i[words[len(words)-k-1]] for k in range(len(words)-j-1)] + [0 for _ in range(j+window_size-len(words)+1)])
                target_set.append(w2i[words[j]])
            else:
                input_set.append([w2i[words[j-k-1]] for k in range(window_size)] + [w2i[words[j+k+1]] for k in range(window_size)])
                target_set.append(w2i[words[j]])
    if mode=="SG":
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
    emb,_ = word2vec_trainer(input_set, target_set, len(w2i), codedict, freqtable, mode=mode, NS=ns, dimension=300, epoch=1, learning_rate=0.01)
    Analogical_Reasoning_Task(emb,w2i,i2w,mode,part,ns)

main()