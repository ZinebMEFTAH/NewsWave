import json
import re
import nltk  # Import nltk library
from nltk.corpus import stopwords  # Import stopwords
from nltk.stem import PorterStemmer # Import stemmer

nltk.download('stopwords') # Download stopwords
ps = PorterStemmer() # Initialize stemmer

# Specify the columns you want to read
columns_to_read = ["headline", "category", "short_description"]  # Replace with your actual column names

# Specify the number of rows to read
num_rows = 100000  # Replace with the desired number of rows

# Read the data line by line and handle potential quote issues
data = []
with open('categoryDataset.json', 'r') as f:
    for line in f:
        # Attempt to fix potential quote issues
        fixed_line = line.replace("\\'", "'")  # Replace escaped single quotes
        try:
            data.append(json.loads(fixed_line))
        except json.JSONDecodeError as e:
            print(f"Skipping line due to JSON error: {e}")

# Convert the list of dictionaries to a DataFrame
data = pd.DataFrame(data, columns=columns_to_read).head(num_rows)

data['news'] = data['headline']

# Select only the 'combined_text' and 'category' columns
data = data[['news', 'category']]

print(data)

corpus = []

for i in range(len(data)):

  customer_review = re.sub('[^a-zA-Z]', ' ',data['news'][i])
  customer_review = customer_review.lower()
  customer_review = customer_review.split()
  clean_review = [ps.stem(word) for word in customer_review if not word in set(stopwords.words('english'))] # Use imported stopwords
  clean_review = ' '.join(clean_review)
  corpus.append(clean_review)

unique_categories = data['category'].unique()
print(unique_categories)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features = 1500, min_df = 3, max_df = 0.6)
X = vectorizer.fit_transform(corpus).toarray()


y = data.iloc[:, 1].values
y

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size = 0.20, random_state = 0)


import torch
import torch.nn as nn
from torch.nn import functional as F


Xtrain_ = torch.from_numpy(X_train).float()
Xtest_ = torch.from_numpy(X_test).float()

ytrain_ = torch.from_numpy(y_train)
ytest_ = torch.from_numpy(y_test)


input_size=1500
output_size=26
hidden_size=1000

class Net(nn.Module):
   def __init__(self):
       super(Net, self).__init__()
       self.fc1 = torch.nn.Linear(input_size, hidden_size)
       self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
       self.fc3 = torch.nn.Linear(hidden_size, output_size)


   def forward(self, X):
       X = torch.relu((self.fc1(X)))
       X = torch.relu((self.fc2(X)))
       X = self.fc3(X)

       return F.log_softmax(X,dim=1)
model = Net()


import torch.optim as optim
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.NLLLoss()

epochs = 100

for epoch in range(epochs):
  optimizer.zero_grad()
  Ypred = model(Xtrain_)
  loss = loss_fn(Ypred,  ytrain_)
  loss.backward()
  optimizer.step()
  print('Epoch',epoch, 'loss',loss.item())


sample = ["The political silence on essential care: A critical issue for society"]
sample = vectorizer.transform(sample).toarray()
result = model(torch.from_numpy(sample).float())
_, predicted = torch.max(result.data, -1)
predicted


# Get distinct categories
distinct_categories = data['category'].unique()

# Print the distinct categories
print("Distinct Categories:")
for category in distinct_categories:
    print(category)

# Count the occurrences of each category
category_counts = data['category'].value_counts()

# Print the category counts
print("\nCategory Counts:")
print(category_counts)

news = data['news'].values
categories  = data['category'].values

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(categories)

x_train, x_test, y_train, y_test = train_test_split(news, labels, test_size=0.2, random_state=42)

tokenizer = Tokenizer(num_words=5000)

tokenizer.fit_on_texts(x_train)

x_train_seq = tokenizer.texts_to_sequences(x_train)
x_test_seq = tokenizer.texts_to_sequences(x_test)

max_sequence_length = 200
x_train_pad = pad_sequences(x_train_seq, maxlen=max_sequence_length)
x_test_pad = pad_sequences(x_test_seq, maxlen=max_sequence_length)

model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=max_sequence_length))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(np.unique(labels)), activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train_pad, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(x_test_pad, y_test)
print(f'Test Accuracy: {accuracy:.4f}')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Convert data to torch tensors
Xtrain_ = torch.from_numpy(x_train_pad).long()
Xtest_ = torch.from_numpy(x_test_pad).long()
ytrain_ = torch.from_numpy(y_train).long()
ytest_ = torch.from_numpy(y_test).long()

# Define the model
class LSTMNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_size):
        super(LSTMNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_size)

    def forward(self, X):
        X = self.embedding(X)
        X, _ = self.lstm(X)
        X = torch.relu(self.fc1(X[:, -1, :]))  # Use the output of the last LSTM cell
        X = self.fc2(X)
        return F.log_softmax(X, dim=1)

# Parameters
vocab_size = 5000
embedding_dim = 128
hidden_dim = 128
output_size = len(np.unique(labels))
epochs = 10

model = LSTMNet(vocab_size, embedding_dim, hidden_dim, output_size)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.NLLLoss()

# Training loop
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    Ypred = model(Xtrain_)
    loss = loss_fn(Ypred, ytrain_)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch}, loss {loss.item()}')

# Evaluate the model
model.eval()
with torch.no_grad():
    Ypred_test = model(Xtest_)
    test_loss = loss_fn(Ypred_test, ytest_)
    _, predicted = torch.max(Ypred_test, 1)
    accuracy = (predicted == ytest_).float().mean()
    print(f'Test Loss: {test_loss.item()}, Test Accuracy: {accuracy.item()}')
