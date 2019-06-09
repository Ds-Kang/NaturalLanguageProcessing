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
            if word not in w2i.keys():  #없는 단어에 대해서는 제외하고 실행
                OOV=1
                break
        if(OOV==1):
            continue
        x=embedding[w2i[words[1]]]-embedding[w2i[words[0]]]+embedding[w2i[words[2]]]


        length = (embedding*embedding).sum(1)**0.5 #cosine similarity 계산 위한 크기
        inputVector = torch.unsqueeze(x,0)/(x*x).sum()**0.5
        sim = (inputVector@embedding.t())[0]/length #cosine similarity 계산
        values, indices = sim.squeeze().topk(5) #상위 5개 저장


        result.write("\n")
        result.write("===============================================\n")
        result.write("The most similar words to \"" + words[1] + "-" + words[0] + "+" + words[2]+ "\"\n")
        result.write("Answer is: "+ words[3]+"\n")
        for ind, val in zip(indices,values):
            result.write(i2w[ind.item()]+":%.3f"%(val,)+"\n")
        result.write("===============================================\n")
        result.write("\n")

    result.close()



def subsampling(input_seq,target_seq,mode):
###############################  Output  #########################################
# subsampled : Subsampled sequence                                               #
##################################################################################

    subsampled_input_seq=[]
    subsampled_target_seq=[]
    t=1e-5 #subsamping 상수 t

    if mode=="CBOW":
        freq=Counter(target_seq) #frequent table 만들기 위해 단어 개수 측정
        for inputs, target in zip(input_seq,target_seq):
            if (1-sqrt(t/freq[target]))>random.random(): #확률에 따른 sequence 값 변경
                subsampled_input_seq.append(inputs)
                subsampled_target_seq.append(target)

    else: #Skip-Gram
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
        if(int(j)==0): #left에 대한 확률과 gradient
            sig=1/(1+torch.exp(-torch.dot(inputMatrix[centerWord],outputMatrix[way])))
            grad_out[way]=(sig-1) * inputMatrix[centerWord]
            grad_in+=(sig-1) * outputMatrix[way]
            loss-=torch.log(sig)
        else: #right에 대한 확률과 gradient
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
    sigmoid = 1/(1+torch.exp(score)) #오답인 경우 sigmoid 함수에 -x값을 대입
    sigmoid[0][0] = 1-sigmoid[0][0] #정답인 경우 올바른 sigmoid 함수값 나타나도록 변경

    loss=torch.sum(-torch.log(sigmoid))
    grad=torch.zeros_like(sigmoid)
    grad[0][0]=(sigmoid[0][0]-1) #정답인 경우의 gradient
    grad[0][1:]=(1-sigmoid[0][1:]) #오답인 경우의 gradient

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
        if(int(j)==0): #left에 대한 확률과 gradient
            sig=1/(1+torch.exp(-torch.dot(torch.mean(inputMatrix[contextWords],0),outputMatrix[way])))
            grad_out[way]=(sig-1) * torch.mean(inputMatrix[contextWords],0,keepdim=True)
            grad_in+=(sig-1) * outputMatrix[way]
            loss-=torch.log(sig)
        else: #right에 대한 확률과 gradient
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
    sigmoid = 1/(1+torch.exp(score)) #오답인 경우 sigmoid 함수에 -x값을 대입
    sigmoid[0][0] = 1-sigmoid[0][0] #정답인 경우 올바른 sigmoid 함수값 나타나도록 변경

    loss=torch.sum(-torch.log(sigmoid))
    grad=torch.zeros_like(sigmoid)
    grad[0][0]=(sigmoid[0][0]-1) #정답인 경우의 gradient
    grad[0][1:]=(1-sigmoid[0][1:]) #오답인 경우의 gradient

    grad_in=torch.mm(grad,outputMatrix)
    grad_out=torch.mm(grad.t(),torch.mean(inputMatrix[contextWords],0,keepdim=True))

    return loss, grad_in, grad_out


def word2vec_trainer(input_seq, target_seq, numwords, codes, stats, mode="CBOW", NS=20, dimension=100, learning_rate=0.025, epoch=3):
# train_seq : list(tuple(int, list(int))

# Xavier initialization of weight matrices
    W_in = torch.randn(numwords, dimension) / (dimension**0.5)
    if NS==0:
        #HS를 사용할 경우 output dimension이 node 개수를 포함해야 하므로 2배 필요하다 판단하고 설정하였다.
        W_out = torch.randn(2*numwords, dimension) / (dimension**0.5)  
        #각 단어마다 왼쪽으로 갈 경우 기존 노드의 way의 2배+1 오른쪽으로 갈 경우 기존 노드의 way의 2배+2가 되도록 way를 설정하였다.
        #이를 그대로 output dimension으로 설정할 경우 2의 n제곱 꼴로 크기가 커지기 때문에 메모리 문제가 발생하여 1~2*numword로 치환해주었다.
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
        #subsampling
        input_seq,target_seq =subsampling(input_seq,target_seq,mode)

        #Training word2vec using SGD(Batch size : 1)
        for inputs, output in zip(input_seq,target_seq):
            i+=1
            if mode=="CBOW":
                if NS==0:
                    #Only use the activated rows of the weight matrix
                    #activated should be torch.tensor(K,) so that activated W_out has the form of torch.tensor(K, D)
                    ways = [] #단어의 way를 tensor 형태로 변환하는 코드
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
                    words=[output] #output의 첫 단어를 맞추는 단어로 나머지를 오답인 단어로 설정
                    for _ in range(NS):
                        while 1:    
                            word = random.choice(stats) #frequent table에서 임의로 단어 추출
                            if word not in words: #중복 단어 추출되지 않도록 설정
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
                    ways = [] #단어의 way를 tensor 형태로 변환하는 코드
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
                    words=[output] #output의 첫 단어를 맞추는 단어로 나머지를 오답인 단어로 설정
                    for _ in range(NS):
                        while 1:    
                            word = random.choice(stats) #frequent table에서 임의로 단어 추출
                            if word not in words: #중복 단어 추출되지 않도록 설정
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