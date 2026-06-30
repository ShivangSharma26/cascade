import torch
import torch.nn as nn
import numpy as np

class UserHistoryLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1):
        super(UserHistoryLSTM, self).__init__()
        # Embedding for item/ad IDs
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        # Final fully connected layer to predict click probability
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        embeds = self.embedding(x)
        # embeds shape: (batch_size, sequence_length, embedding_dim)
        
        lstm_out, (hidden, cell) = self.lstm(embeds)
        # hidden[-1] is the last hidden state of the sequence
        last_hidden = hidden[-1]
        
        out = self.fc(last_hidden)
        return self.sigmoid(out)

def train_lstm(model, dataloader, epochs=5, lr=0.001):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for seqs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(seqs).squeeze()
            loss = criterion(outputs, targets.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"LSTM Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
    return model

if __name__ == '__main__':
    print("LSTM module ready.")
