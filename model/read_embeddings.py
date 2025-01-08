import pickle

def load_embeddings(filename="embeddings.pkl"):
    with open(filename, 'rb') as f:
        embeddings, labels = pickle.load(f)
    return embeddings, labels

# Usage
embeddings, labels = load_embeddings()
print("Loaded embeddings:", embeddings)
print("Loaded labels:", labels)
