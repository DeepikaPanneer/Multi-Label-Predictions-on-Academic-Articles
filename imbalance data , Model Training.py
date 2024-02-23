#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import random
train_data = pd.read_csv('train.csv')

# Preprocessing (Update as per your dataset specifics)
train_data['TEXT'] = train_data['TITLE'] + ' ' + train_data['ABSTRACT']
X = train_data['TEXT']
y = train_data[['Computer Science', 'Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance']]

# Split the dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
X_val_tfidf = vectorizer.transform(X_val).toarray()


# In[7]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import random

# Load the dataset
train_data = pd.read_csv('train.csv')
train_data['TEXT'] = train_data['TITLE'] + ' ' + train_data['ABSTRACT']

X = train_data['TEXT']
y = train_data[['Computer Science', 'Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance']].values

# Split the dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
X_val_tfidf = vectorizer.transform(X_val).toarray()

# Helper function to identify minority labels
def get_tail_labels(y):
    tail_labels = [i for i in range(y.shape[1]) if np.sum(y[:, i]) < (y.shape[0] / 2)]
    return tail_labels

# class distribution before applying dynamic MLSMOTE
print("Class distribution before applying dynamic MLSMOTE:")
for i, label in enumerate(['Computer Science', 'Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance']):
    print(f"{label}: {np.sum(y_train[:, i])}")

# Dynamic MLSMOTE function
def dynamic_MLSMOTE(X, y, target_balance=4500):
    n_neighbors = min(5, len(X) - 1)
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(X)
    tail_labels = get_tail_labels(y)
    synthetic_samples = []
    synthetic_labels = []

    for i in tail_labels:
        current_count = np.sum(y[:, i])
        n_samples = max(target_balance - current_count, 0)  # Calculate the number of samples to generate
        target_indices = np.where(y[:, i] == 1)[0]
        
        if len(target_indices) >= n_neighbors:
            nn = neigh.kneighbors(X[target_indices], return_distance=False)
            for _ in range(n_samples):
                sample_index = random.choice(range(len(target_indices)))
                nn_indices = nn[sample_index, 1:]
                chosen_nn = random.choice(nn_indices)
                step = np.random.rand()
                synthetic_sample = X[target_indices[sample_index]] + step * (X[chosen_nn] - X[target_indices[sample_index]])
                synthetic_samples.append(synthetic_sample)
                synthetic_label = y[target_indices[sample_index]].copy()
                synthetic_labels.append(synthetic_label)
    
    if len(synthetic_samples) > 0:
        X_synthetic = np.vstack(synthetic_samples)
        y_synthetic = np.vstack(synthetic_labels)
        X_balanced = np.vstack((X, X_synthetic))
        y_balanced = np.vstack((y, y_synthetic))
        return X_balanced, y_balanced
    else:
        return X, y

# Convert y_train to numpy array for processing
y_train_np = y_train

# Adjust this target balance
target_balance = 4500  
X_balanced_tfidf, y_balanced = dynamic_MLSMOTE(X_train_tfidf, y_train_np, target_balance=target_balance)

# class distribution after applying dynamic MLSMOTE
print("/n")
print("Class distribution after applying dynamic MLSMOTE:")
for i, label in enumerate(['Computer Science', 'Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance']):
    print(f"{label}: {np.sum(y_balanced[:, i])}")


# In[8]:



# X_balanced_tfidf and y_balanced arebalanced datasets returned from dynamic_MLSMOTE
feature_names = vectorizer.get_feature_names_out()
df_features = pd.DataFrame(X_balanced_tfidf, columns=feature_names)
df_labels = pd.DataFrame(y_balanced, columns=['Computer Science', 'Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance'])
df_balanced = pd.concat([df_features, df_labels], axis=1)


# In[9]:


import matplotlib.pyplot as plt

label_sums = df_balanced.iloc[:, -6:].sum()
# bar plot for label distribution
label_sums.plot(kind='bar')
plt.title('Class Distribution After Applying Dynamic MLSMOTE')
plt.ylabel('Number of Samples')
plt.xlabel('Class Labels')
plt.xticks(rotation=45)
plt.show()


# In[10]:


from sklearn.manifold import TSNE

#TSNE for dimensionality reduction to visualize high-dimensional data
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(df_features.iloc[:1000])

plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
plt.title('t-SNE visualization of the Balanced Dataset')
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
plt.show()


# In[12]:


from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

X_test_tfidf = vectorizer.transform(X_val).toarray()
#train the model on the balanced dataset
model = MultiOutputClassifier(LogisticRegression(solver='lbfgs', max_iter=1000))
model.fit(X_balanced_tfidf, y_balanced)  
y_pred = model.predict(X_val_tfidf) 

# Evaluate the model on the validation set
accuracy = accuracy_score(y_val, y_pred)  
f1 = f1_score(y_val, y_pred, average='micro')  

print(f'Accuracy: {accuracy:.4f}')
print(f'F1 Score: {f1:.4f}')


# In[ ]:


from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset
from skmultilearn.adapt import MLkNN
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, hamming_loss
from scipy.stats import gmean
from sklearn.preprocessing import MultiLabelBinarizer

# Binarize the labels for metrics calculation
mlb = MultiLabelBinarizer()
y_val_binarized = mlb.fit_transform(y_val)

# Define and train models with transformation methods
transformations = {
    'Binary Relevance': BinaryRelevance,
    'Classifier Chains': ClassifierChain,
    'Label Powerset': LabelPowerset
}

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'MLkNN': MLkNN(k=3),
    'Decision Tree': DecisionTreeClassifier()
}

for transformation_name, Transformation in transformations.items():
    for model_name, model in models.items():
        if model_name != 'MLkNN':  # MLkNN requires sparse format
            classifier = Transformation(classifier=model)
            classifier.fit(X_balanced_tfidf, y_balanced)
            predictions = classifier.predict(X_val_tfidf)
            predictions = mlb.inverse_transform(predictions)
        else:
            classifier = model
            classifier.fit(X_balanced_tfidf, y_balanced)
            predictions = classifier.predict(X_val_tfidf)
            predictions = mlb.inverse_transform(predictions)
        
        # Calculate metrics
        y_pred_binarized = mlb.transform(predictions)
        accuracy = accuracy_score(y_val_binarized, y_pred_binarized)
        f1 = f1_score(y_val_binarized, y_pred_binarized, average='micro')
        precision = precision_score(y_val_binarized, y_pred_binarized, average='micro')
        recall = recall_score(y_val_binarized, y_pred_binarized, average='micro')
        gmean_val = gmean(y_val_binarized[y_val_binarized.sum(axis=1) != 0], y_pred_binarized[y_val_binarized.sum(axis=1) != 0], axis=None)  # Adjusted for binary relevance
        hamming = hamming_loss(y_val_binarized, y_pred_binarized)
        
        print(f"{transformation_name} with {model_name}: Accuracy: {accuracy}, F1: {f1}, Precision: {precision}, Recall: {recall}, G-mean: {gmean_val}, Hamming Loss: {hamming}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
train_data = pd.read_csv('train.csv')

# Preprocessing (Update as per your dataset specifics)
X = train_data['TEXT']
y = train_data[['Computer Science', 'Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance']]

# Split the dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
X_val_tfidf = vectorizer.transform(X_val).toarray()

# Convert to PyTorch tensors
X_train_torch = torch.tensor(X_train_tfidf, dtype=torch.float32)
X_val_torch = torch.tensor(X_val_tfidf, dtype=torch.float32)
y_train_torch = torch.tensor(y_train.values, dtype=torch.float32)
y_val_torch = torch.tensor(y_val.values, dtype=torch.float32)

# Dataset
class ResearchPapersDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = ResearchPapersDataset(X_train_torch, y_train_torch)
val_dataset = ResearchPapersDataset(X_val_torch, y_val_torch)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model
class MultiLabelNN(nn.Module):
    def __init__(self):
        super(MultiLabelNN, self).__init__()
        self.fc1 = nn.Linear(10000, 2048)  # Increased model capacity
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Added dropout for regularization
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 6)  # 6 classes
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

model = MultiLabelNN()

# Loss function
loss_fn = nn.BCEWithLogitsLoss()  # Using BCEWithLogitsLoss for multi-label classification

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Training
num_epochs = 10  # Increased epochs for better convergence
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = loss_fn(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step()
    
    # Validation accuracy and F1 score after each epoch
    model.eval()
    with torch.no_grad():
        y_pred = []
        y_true = []
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            predictions = torch.sigmoid(outputs).round().numpy()
            y_pred.extend(predictions)
            y_true.extend(y_batch.numpy())
    
    val_accuracy = accuracy_score(y_true, y_pred)
    val_f1 = f1_score(y_true, y_pred, average='micro')
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}, Validation Accuracy: {val_accuracy}, Validation F1 Score: {val_f1}')


# In[ ]:




