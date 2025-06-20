🔤 Word Predictor using LSTM in PyTorch
This project demonstrates a Word Prediction Model built using an LSTM (Long Short-Term Memory) neural network in PyTorch. The model is trained on a sequence of words and learns to predict the next word based on the given context, showcasing the fundamentals of language modeling.

🧠 Objective
To build a language model that takes a sequence of words as input and predicts the next likely word using LSTM, learning from text patterns in the dataset.

📚 Dataset

NLTK or spaCy (for tokenization)

📦 Project Structure

word-predictor-lstm/
│
├── data.txt                  # Training corpus
├── preprocess.py             # Text preprocessing and vocabulary creation
├── train.py                  # Model training script
├── model.py                  # LSTM model class
├── predict.py                # Word prediction using trained model
├── utils.py                  # Helper functions
└── README.md
⚙️ How It Works
🧹 1. Preprocessing
Tokenize text into words

Build vocabulary (word-to-index and index-to-word mappings)

Create input-output pairs using a sliding window:

Input: ["The", "sun", "rises"] → Output: "in"

🧠 2. LSTM Model Architecture
class WordPredictor(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(WordPredictor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

🔁 3. Training
Loss Function: CrossEntropyLoss

Optimizer: Adam

Trained on word sequences for multiple epochs

🔮 4. Prediction
Input a sequence like: "The sun rises"

Model predicts: "in"


