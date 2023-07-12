import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Define the Encoder model
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input_seq, hidden):
        embedded = self.embedding(input_seq)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

# Define the Decoder model
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_seq, hidden):
        embedded = self.embedding(input_seq)
        output, hidden = self.gru(embedded, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

# Define the Seq2Seq model that combines the Encoder and Decoder
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, input_seq, target_seq, teacher_forcing_ratio=0.5):
        batch_size = input_seq.size(0)
        target_len = target_seq.size(1)
        target_vocab_size = self.decoder.out.out_features

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(self.device)
        encoder_output, hidden = self.encoder(input_seq, torch.zeros(1, batch_size, self.encoder.hidden_size).to(self.device))

        decoder_input = torch.tensor([[SOS_token]] * batch_size).to(self.device)
        
        for t in range(1, target_len):
            output, hidden = self.decoder(decoder_input, hidden)
            outputs[t] = output
            teacher_force = torch.rand(1) < teacher_forcing_ratio
            top1 = output.max(1)[1]
            decoder_input = target_seq[:, t] if teacher_force else top1

        return outputs

# Define the dataset class
class TranslationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

# Define the training function
def train(model, optimizer, criterion, input_seq, target_seq):
    optimizer.zero_grad()

    input_seq = input_seq.to(device)
    target_seq = target_seq.to(device)

    output = model(input_seq, target_seq)
    output = output[1:].view(-1, output.shape[-1])
    target_seq = target_seq[:, 1:].reshape(-1)

    loss = criterion(output, target_seq)
    loss.backward()

    optimizer.step()

    return loss.item()

# Define the main function
def main():
    # Define hyperparameters
    input_size = len(input_vocab)
    output_size = len(output_vocab)
    hidden_size = 256
    learning_rate = 0.01
    batch_size = 32
    num_epochs = 10
    teacher_forcing_ratio = 0.5

    # Prepare the dataset
    dataset = TranslationDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the models
    encoder = Encoder(input_size, hidden_size)
    decoder = Decoder(hidden_size, output_size)
    model = Seq2Seq(encoder, decoder, device).to(device)

    # Define the loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        for i, (input_seq, target_seq) in enumerate(dataloader):
            loss = train(model, optimizer, criterion, input_seq, target_seq)
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), 'translation_model.pt')

# Run the main function
if __name__ == '__main__':
    main()
