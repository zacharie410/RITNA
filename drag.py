import torch
import torch.nn as nn
import wikipediaapi

def get_wikipedia_summary(page_title):
    wiki_wiki = wikipediaapi.Wikipedia('RAG_EDUCATIONAL_DEMONSTRATION')
    page = wiki_wiki.page(page_title)
    return page.summary if page.exists() else None

class DynamicTokenizer:
    def __init__(self):
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self.inverse_vocab = {0: "<PAD>", 1: "<UNK>"}

    def update_vocab(self, text):
        for word in text.split():
            if word not in self.vocab:
                index = len(self.vocab)
                self.vocab[word] = index
                self.inverse_vocab[index] = word

    def encode(self, text):
        return [self.vocab.get(word, self.vocab["<UNK>"]) for word in text.split()]

    def decode(self, indices):
        return " ".join(self.inverse_vocab.get(index, "<UNK>") for index in indices)

class SimpleQA(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64):
        super(SimpleQA, self).__init__()
        self.embedding_dim = embedding_dim
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=2, batch_first=True),
            num_layers=1
        )
        self.out = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)  # Embed token indices
        x = self.transformer(x)  # Pass through transformer
        return self.out(x)  # Output layer for each token

    def expand_vocab(self, new_vocab_size):
        new_embed = nn.Embedding(new_vocab_size, self.embedding_dim)
        new_out = nn.Linear(self.embedding_dim, new_vocab_size)
        new_embed.weight.data[:self.embed.num_embeddings] = self.embed.weight.data
        new_out.weight.data[:self.out.out_features] = self.out.weight.data
        new_out.bias.data[:self.out.out_features] = self.out.bias.data
        self.embed = new_embed
        self.out = new_out

def train_and_query(model, tokenizer, query, document_summary, epochs=10):
    tokenizer.update_vocab(document_summary)
    tokenizer.update_vocab(query)
    vocab_size = len(tokenizer.vocab)
    model.expand_vocab(vocab_size)

    input_text = tokenizer.encode(document_summary + " " + query)
    target_text = tokenizer.encode(document_summary)
    
    # Ensure input and target are the same length for training
    max_length = max(len(input_text), len(target_text))
    input_text += [tokenizer.vocab["<PAD>"]] * (max_length - len(input_text))
    target_text += [tokenizer.vocab["<PAD>"]] * (max_length - len(target_text))

    input_tensor = torch.tensor([input_text], dtype=torch.long)
    target_tensor = torch.tensor([target_text], dtype=torch.long)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab["<PAD>"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = criterion(output.view(-1, vocab_size), target_tensor.view(-1))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}: Loss = {loss.item()}")

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        predicted_indices = output.argmax(-1).tolist()
        # Flatten the list of lists to a single list
        predicted_indices = [idx for sublist in predicted_indices for idx in sublist]
        print("Predicted Answer:", tokenizer.decode(predicted_indices))

# Example use
tokenizer = DynamicTokenizer()
model = SimpleQA(len(tokenizer.vocab))
summary = get_wikipedia_summary("Albert Einstein")
train_and_query(model, tokenizer, "What is Einstein known for?", summary)
