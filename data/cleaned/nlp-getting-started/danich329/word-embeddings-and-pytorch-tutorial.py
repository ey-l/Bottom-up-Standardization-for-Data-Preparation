import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
training_data = pd.read_csv('data/input/nlp-getting-started/train.csv')
training_data.head()
import gensim
training_data.text.apply(lambda x: x.split(' ')).head().values
training_corpus = training_data.text.apply(lambda x: x.split(' '))
model = gensim.models.Word2Vec(sentences=training_corpus)
model.wv.vocab
model.wv['Forest']
model.wv.most_similar('Forest')
model.wv.vectors

class MyBackpack:
    pass

class MyBackPack:

    def __init__(self):
        self.container = []

    def hold(self, obj):
        self.container.append(obj)
backpack = MyBackPack()
backpack.hold('candy')
print(backpack.container)

class Container:

    def __init__(self):
        self.message = 'I am a container whoo.'

class MyBackPack(Container):

    def __init__(self):
        super().__init__()
        self.container = []

    def hold(self, obj):
        self.container.append(obj)
backpack = MyBackPack()
backpack.hold('candy')
print(backpack.message)
import torch
import torch.nn as nn

class FirstNeuralNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        pass

class FirstNeuralNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)

    def forward(self, data_input):
        output_layer1 = self.layer1(data_input)
        output_layer1 = self.relu(output_layer1)
        output_layer2 = self.layer2(output_layer1)
        return output_layer2
first_neural_network = FirstNeuralNetwork(300, 100, 2)
print(first_neural_network)
sentence_length = training_data.text.apply(lambda x: x.split(' ')).apply(len).max()
sentence_length
word_map = {word: idx for (idx, word) in enumerate(model.wv.vocab, start=2)}
word_map
training_data.text.apply(lambda x: x.split(' '))[0]
training_sentence_data = training_data.text.apply(lambda x: list(map(lambda word: word_map[word] if word in word_map else 1, x.split(' '))))
print(training_sentence_data[0])
training_sentence_data = list(map(lambda x: pd.np.pad(x, (0, sentence_length - len(x))), training_sentence_data))
training_sentence_data[0]
torch.LongTensor(training_sentence_data[0])
training_sentence_data = list(map(lambda x: torch.LongTensor(x), training_sentence_data))
training_sentence_data[0]
pd.np.random.seed(100)
word_vectors_for_training = pd.np.insert(model.wv.vectors, 0, pd.np.random.uniform(model.wv.vectors.min(), model.wv.vectors.max(), 100), axis=0)
word_vectors_for_training = pd.np.insert(word_vectors_for_training, 0, pd.np.zeros(100), axis=0)
word_vectors_for_training = torch.FloatTensor(word_vectors_for_training)
word_vectors_for_training

class FirstNeuralNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.embedding_layer = nn.EmbeddingBag.from_pretrained(word_vectors_for_training, mode='mean')
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)

    def forward(self, data_input):
        embedded_data_input = self.embedding_layer(data_input)
        output_layer1 = self.layer1(embedded_data_input)
        output_layer1 = self.relu(output_layer1)
        output_layer2 = self.layer2(output_layer1)
        return output_layer2.squeeze()
training_sentence_data = torch.stack(training_sentence_data)
first_neural_network = FirstNeuralNetwork(100, 50, 1)
import torch.utils.data as data
dataset = data.TensorDataset(training_sentence_data, torch.FloatTensor(training_data.target.values))
dataloader = data.DataLoader(dataset, batch_size=256)
import torch.optim as optim
optimizer = optim.Adam(first_neural_network.parameters(), lr=0.001)
loss_fn = nn.BCEWithLogitsLoss()
import tqdm
for n in tqdm.tqdm(range(100)):
    avg_loss = []
    for batch in dataloader:
        optimizer.zero_grad()
        output = first_neural_network(batch[0])
        loss = loss_fn(output, batch[1])
        avg_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    print(pd.np.mean(avg_loss))