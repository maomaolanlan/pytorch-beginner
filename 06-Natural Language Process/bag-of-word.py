__author__ = 'SherlockLiao'

import torch
from torch import nn, optim
import torch.nn.functional as F

# The Continuous Bag-of-Words model (CBOW) is frequently used in NLP deep learning. 
# It is a model that tries to predict words given the context of a few words before
# and a few words after the target word. This is distinct from language modeling, 
# since CBOW is not sequential and does not have to be probabilistic. 
# Typcially, CBOW is used to quickly train word embeddings, 
# and these embeddings are used to initialize the embeddings of some more complicated model. 
# Usually, this is referred to as pretraining embeddings. 
# It almost always helps performance a couple of percent.

# 从上下文来预测一个文字
# http://www.pytorchtutorial.com/10-minute-pytorch-8/

CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
raw_text     = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

vocab = set(raw_text)
word_to_idx = {word: i for i, word in enumerate(vocab)}

# 使用raw_text生产训练所需要的数据，结构(context : target)
data = []
for i in range(CONTEXT_SIZE, len(raw_text) - CONTEXT_SIZE):
    context = [
        raw_text[i - 2], raw_text[i - 1], raw_text[i + 1], raw_text[i + 2]
    ]
    target = raw_text[i]
    data.append((context, target))


class CBOW(nn.Module):
    def __init__(self, n_word, n_dim, context_size):
        super(CBOW, self).__init__()
        #word2vec
        self.embedding = nn.Embedding(n_word, n_dim)
        self.project = nn.Linear(n_dim, n_dim, bias=False)
        self.linear1 = nn.Linear(n_dim, 128)
        self.linear2 = nn.Linear(128, n_word)

    def forward(self, x):
        x = self.embedding(x)
        x = self.project(x)
        x = torch.sum(x, 0, keepdim=True)
        x = self.linear1(x)
        x = F.relu(x, inplace=True)
        x = self.linear2(x)
        x = F.log_softmax(x,dim=1)
        return x


model = CBOW(len(word_to_idx), 100, CONTEXT_SIZE)
if torch.cuda.is_available():
    model = model.cuda()

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

for epoch in range(100):
    print('epoch {}'.format(epoch+1))
    print('*' * 20)
    running_loss = 0
    eval_acc = 0.
    for word in data:
        context, target = word
        context = torch.LongTensor([word_to_idx[i] for i in context])
        target = torch.LongTensor([word_to_idx[target]])
        if torch.cuda.is_available():
            context = context.cuda()
            target = target.cuda()
        # forward
        out = model(context)
        
        #最大值下标即为预测值
        _, pred = torch.max(out, 1)
        num_correct = (pred == target).sum()
        eval_acc += num_correct.item()
        
        loss = criterion(out, target)
        running_loss += loss.item()
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('loss: {:.6f}, Acc: {:.6f}'.format(running_loss / len(data)
        , eval_acc / (len(data))))
