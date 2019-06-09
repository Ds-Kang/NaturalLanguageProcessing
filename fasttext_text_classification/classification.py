import csv
import torch

def FNV1a_hashing(bigram):
    FNV_prime=16777619
    hashe = 2166136261 #내장함수와 구별위해
    for char in bigram:
        hashe = hashe ^ ord(char)
        hashe = hashe * FNV_prime
    return hashe

def loss(ans, input_text, W_out):
    score=torch.mm(input_text,W_out.t())
    e = torch.exp(score)
    softmax = e / torch.sum(e, dim=1, keepdim=True)
    loss=-torch.log(softmax[0][ans])

    grad=softmax
    grad[0][ans]=grad[0][ans]-1

    grad_in=torch.mm(grad,W_out).t()
    grad_out=torch.mm(grad.t(),input_text)

    return loss,grad_in, grad_out


train_csv = open('ag_news_csv/train.csv', 'r')
train = csv.reader(train_csv)

dimension=10
learning_rate=0.05
W_in = torch.randn(100000, dimension) / (dimension**0.5)
W_out = torch.randn(4, dimension) / (dimension**0.5)

cnt=0
epoch=1
for _ in range(epoch):
    for line in train:
        words=line[2].split()
        bi_i=[]
        for i in range(len(words)-1):
            bi_i.append(FNV1a_hashing('{}_{}'.format(words[i],words[i+1]))%100000)

        input_text=torch.mean(W_in[bi_i],0,keepdim=True)
        L, G_in, G_out = loss(int(line[0])-1,input_text,W_out)

        W_in[bi_i] -= learning_rate*G_in.squeeze()/len(bi_i)
        W_out -= learning_rate*G_out

        if cnt%5000==0:
            print(L)
        cnt+=1

test_csv = open('ag_news_csv/test.csv', 'r')
test = csv.reader(test_csv)

output=open('output.txt','w')

ans_cnt=0
cnt=0
for line in test:
    words=line[2].split()
    bi_i=[]
    for i in range(len(words)-1):
        bi_i.append(FNV1a_hashing('{}_{}'.format(words[i],words[i+1]))%100000)

    input_text=torch.mean(W_in[bi_i],0,keepdim=True)
    ans=torch.argmax(torch.mm(input_text,W_out.t()))+1

    classes={1:'World',2:'Sports',3:'Business',4:'Sci/Tech'}
    output.write('Prediction of index '+str(cnt)+' is '+classes[int(ans)]+'\n')
    if(ans==int(line[0])):
        output.write('Answer is '+classes[int(line[0])]+' -> Correct :)\n\n')
        ans_cnt+=1
    else:
        output.write('Answer is '+classes[int(line[0])]+' -> Wrong :(\n\n')

    cnt+=1
    
output.write('Probability of Correctness: %f'%(ans_cnt/cnt,))

train_csv.close()
test_csv.close()
output.close()


