import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])  # use last time step
        return out.squeeze()

def train_lstm_model(X, y, seq_length=5, epochs=50, lr=0.001):
    # Normalize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create sequences
    sequences = []
    targets = []
    for i in range(len(X) - seq_length):
        sequences.append(X_scaled[i:i+seq_length])
        targets.append(y[i+seq_length])

    X_tensor = torch.tensor(np.array(sequences), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(targets), dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    # Model
    model = LSTMModel(input_size=X.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in dataloader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
    
    return model, scaler, seq_length
