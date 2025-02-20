import json
import re
import nltk  # Import nltk library
from nltk.corpus import stopwords  # Import stopwords
from nltk.stem import PorterStemmer # Import stemmer
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

nltk.download('stopwords') # Download stopwords
ps = PorterStemmer() # Initialize stemmer

# Specify the columns you want to read
columns_to_read = ["headline", "category", "short_description"]  # Replace with your actual column names


# Convert the list of dictionaries to a DataFrame
data = pd.read_csv("train.csv")

data['category'] = data['Class Index']

data['news'] = data['Description']

# Select only the 'combined_text' and 'category' columns
data = data[['category', 'Description']]

print(data)

corpus = []

for i in range(len(data)):
    customer_review = re.sub('[^a-zA-Z]', ' ', data['news'][i])
    customer_review = customer_review.lower()
    customer_review = customer_review.split()
    clean_review = [ps.stem(word) for word in customer_review if not word in set(stopwords.words('english'))]  # Use imported stopwords
    clean_review = ' '.join(clean_review)
    corpus.append(clean_review)

unique_categories = data['category'].unique()
print(unique_categories)


vectorizer = TfidfVectorizer(max_features=1500, min_df=3, max_df=0.6)
X = vectorizer.fit_transform(corpus).toarray()

y = data.iloc[:, 1].values
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.20, random_state=0)

Xtrain_ = torch.from_numpy(X_train).float()
Xtest_ = torch.from_numpy(X_test).float()

ytrain_ = torch.from_numpy(y_train).long()
ytest_ = torch.from_numpy(y_test).long()

input_size = 1500
output_size = len(unique_categories)  # Output size should match the number of unique categories
hidden_size = 1000

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, X):
        X = torch.relu(self.fc1(X))
        X = torch.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)

model = Net()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.NLLLoss()

epochs = 100

for epoch in range(epochs):
    optimizer.zero_grad()
    Ypred = model(Xtrain_)
    loss = loss_fn(Ypred, ytrain_)
    loss.backward()
    optimizer.step()
    print('Epoch', epoch, 'loss', loss.item())

sample = ["The political silence on essential care: A critical issue for society"]
sample = vectorizer.transform(sample).toarray()
result = model(torch.from_numpy(sample).float())
_, predicted = torch.max(result.data, -1)
predicted_category = label_encoder.inverse_transform(predicted.numpy())
print(f'The predicted category is: {predicted_category[0]}')