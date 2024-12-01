from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


from transformers import (
    BertTokenizer,
    PreTrainedTokenizer,
)

import argparse

# Command-line argument parser setup
parser = argparse.ArgumentParser(description="Train a neural network")
parser.add_argument("--epochs", type=int, default=100, help="Number of Epochs")
parser.add_argument(
    "--learning_optimization",
    type=str,
    default="default",
    help="Type of Learning Strategy",
)
parser.add_argument("--clean", type=str, default="original", help="Cleaning Level")
parser.add_argument("--stop_early", action=argparse.BooleanOptionalAction)

parser.add_argument(
    "--embedding_dim", type=int, default=128, help="Embedding Dimension"
)
parser.add_argument("--embedding_layers", type=int, default=1, help="Embedding Layers")
parser.add_argument("--comparison_type", type=str, default=None, help="Comparison Type")
parser.add_argument(
    "--tokenizer",
    type=str,
    default="bert-base-cased",
    help="Tokenizer <https://github.com/AkariAsai/pytorch-pretrained-BERT/blob/master/README.md>",
)
args = parser.parse_args()

# Load datasets based on the cleaning level specified in arguments
if args.clean == "original":
    train_df = pd.read_parquet("./data/preprocessed/train/train_raw.parquet")
    test_df = pd.read_parquet("./data/preprocessed/test/test_raw.parquet")
elif args.clean == "a1":
    train_df = pd.read_parquet("./data/preprocessed/train/train_a1.parquet")
    test_df = pd.read_parquet("./data/preprocessed/test/test_a1.parquet")
elif args.clean == "a2":
    train_df = pd.read_parquet("./data/preprocessed/train/train_a2.parquet")
    test_df = pd.read_parquet("./data/preprocessed/test/test_a2.parquet")
else:
    raise ValueError(f"Unexpected --clean data level: {args.clean}")

# Device setup for running on MacbookPro
DEVICE = torch.device("mps")

# Make outccomes deterministic
RANDOM_SEED = 1234
torch.manual_seed(RANDOM_SEED)

tokenizer = BertTokenizer.from_pretrained(args.tokenizer)

# Set up model parameters
VOCAB_SIZE = tokenizer.vocab_size  # From the tokenizer
EMBEDDING_DIM = args.embedding_dim
EMBEDDING_LAYERS = args.embedding_layers
HIDDEN_DIM = EMBEDDING_DIM * 2
OUTPUT_DIM = 1
EPOCHS = args.epochs
BATCH_SIZE = 256

MODEL = "TwitterDisasterRNN"

# Model name for versioning
MODEL_NAME = f"v1.{MODEL}.data_level_{args.clean}.tokenizer_{args.tokenizer}.embedding_dim_{EMBEDDING_DIM}.embedding_layers_{EMBEDDING_LAYERS}.hidden_dim_{HIDDEN_DIM}.comparison_type_{args.comparison_type}.learning_optimization_{args.learning_optimization}.epochs_{EPOCHS}"
print(f"Running: {MODEL_NAME}...")


# Function to calculate the dynamic max_len
def get_max_len_from_data(texts, tokenizer, percentile=99):
    lengths = []
    for text in texts:
        if text is not None:
            encoded = tokenizer(
                text, truncation=False, padding=False, return_tensors="pt"
            )
            lengths.append(encoded["input_ids"].size(1))  # Length of tokenized sequence

    max_len = int(np.percentile(lengths, percentile))

    return max_len


TEXT_MAX_LEN = get_max_len_from_data(train_df["text"].to_list(), tokenizer)
LOCATION_MAX_LEN = get_max_len_from_data(train_df["location"].to_list(), tokenizer)
KEYWORD_MAX_LEN = get_max_len_from_data(train_df["keyword"].to_list(), tokenizer)


def preprocess_dataframe(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizer,
    text_max_length: int,
    keyword_max_length: int,
    location_max_length: int,
):
    ids, tokens, attentions, targets = [], [], [], []

    max_length = text_max_length + keyword_max_length + location_max_length

    df["keyword"] = df["keyword"].fillna("")
    df["location"] = df["location"].fillna("")

    for _, row in df.iterrows():
        combined_text = (
            f"[SEP] {row['text']} [SEP] {row['keyword']} [SEP] {row['location']}"
        )
        tokenized = tokenizer(
            combined_text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        these_tokens = tokenized["input_ids"].tolist()[0]
        attention_mask = tokenized["attention_mask"].tolist()[0]

        ids.append(row["id"])
        tokens.append(these_tokens)
        attentions.append(attention_mask)
        try:
            targets.append(row["target"])
        except KeyError:
            continue

    if len(targets) > 0:
        return pd.DataFrame(
            {"id": ids, "tokens": tokens, "attention": attentions, "target": targets}
        )
    else:
        return pd.DataFrame({"id": ids, "tokens": tokens, "attention": attentions})


train_tokens_df = preprocess_dataframe(
    train_df, tokenizer, TEXT_MAX_LEN, KEYWORD_MAX_LEN, LOCATION_MAX_LEN
)


test_tokens_df = preprocess_dataframe(
    test_df, tokenizer, TEXT_MAX_LEN, KEYWORD_MAX_LEN, LOCATION_MAX_LEN
)


class TwitterCombinedDataset(Dataset):
    def __init__(self, preprocessed_df):
        self.data = preprocessed_df
        self.has_target = False
        if "target" in list(preprocessed_df.columns):
            self.has_target = True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        result = {
            "id": row["id"],
            "tokens": torch.tensor(row["tokens"], dtype=torch.long),
            "attention": torch.tensor(row["attention"], dtype=torch.long),
        }
        if self.has_target:
            return {**result, "target": torch.tensor(row["target"], dtype=torch.float)}
        else:
            return result


train_combined_df, val_combined_df = train_test_split(
    train_tokens_df, test_size=0.2, random_state=RANDOM_SEED
)

train_combined_dataset = TwitterCombinedDataset(train_combined_df)
val_combined_dataset = TwitterCombinedDataset(val_combined_df)
test_combined_dataset = TwitterCombinedDataset(test_tokens_df)

train_combined_loader = DataLoader(
    train_combined_dataset, batch_size=BATCH_SIZE, shuffle=True
)
val_combined_loader = DataLoader(val_combined_dataset, batch_size=BATCH_SIZE)
test_combined_loader = DataLoader(test_combined_dataset, batch_size=BATCH_SIZE)


class TwitterDisasterRNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        use_attention=True,
        embedding_dim=128,
        embedding_layers=2,
        hidden_dim=256,
        output_dim=1,
        dropout_prob=0.5,
    ):
        super(TwitterDisasterRNN, self).__init__()
        self.use_attention = use_attention
        self.text_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            num_layers=embedding_layers,
            bidirectional=True,
            dropout=dropout_prob,  # Dropout between LSTM layers
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # Account for bidirectional
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(
        self,
        text_tokens,
        text_attention_mask,
    ):
        # Embedding layers
        text_emb = self.text_embedding(text_tokens)

        if self.use_attention is True:
            text_emb = text_emb * text_attention_mask.unsqueeze(-1)

        # Pass through LSTM
        lstm_out, _ = self.lstm(text_emb)

        pooled_output = lstm_out.mean(dim=1)

        pooled_output = self.dropout(pooled_output)

        # Fully connected layer
        logits = self.fc(pooled_output)
        return logits


def train_model(
    model,
    train_loader,
    val_loader,
    zero_weight,
    one_weight,
    epochs,
):
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.BCELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
    )

    if args.learning_optimization == "medium":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=5,
            factor=0.3,
        )

    if args.learning_optimization == "agressive":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=3,
            factor=0.5,
        )

    train_stats = []
    validation_loss_tracker = []

    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        train_loss = 0.0

        all_targets = []
        all_predictions = []

        for batch in train_loader:
            optimizer.zero_grad()
            text_tokens = batch["tokens"].to(DEVICE)
            text_attention_mask = batch["attention"].to(DEVICE)
            targets = batch["target"].to(DEVICE).float()
            all_targets.extend(batch["target"])

            weights = torch.where(targets == 0, zero_weight, one_weight).to(DEVICE)

            outputs = model.forward(
                text_tokens,
                text_attention_mask,
            )

            all_predictions.extend((outputs > 0.5).cpu().numpy())

            loss = criterion(outputs.squeeze(1), targets)

            weighted_loss = loss * weights  # Apply weights to the loss

            weighted_loss.mean().backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += weighted_loss.mean().item()

        elapsed = time.time() - start
        this_learning_rate = optimizer.param_groups[0]["lr"]
        train_f1 = f1_score(all_targets, all_predictions)

        # Validate the model
        validate_stats = validate_model(model, val_loader, criterion)
        val_loss = validate_stats["Validation Loss"]
        val_f1 = validate_stats["Validation F1 Score"]
        val_acc = validate_stats["Validation Accuracy Score"]
        validation_loss_tracker.append(val_loss)

        print(
            (
                f"Epoch {epoch:03d}/{epochs:03d}, "
                f"Learning Rate: {this_learning_rate}, "
                f"Training F1 Score: {train_f1}, "
                f"Val F1 Score: {val_f1}, "
                f"Accuracy: {val_acc}"
            )
        )

        scheduler.step(val_f1)

        these_stats = {
            **validate_stats,
            "Learning Rate": this_learning_rate,
            "Training Loss": train_loss / len(train_loader),
            "Training F1 Score": train_f1,
            "Epoch": epoch,
            "Specified Epochs": epochs,
            "Compute Time": elapsed,
            "Vocab Size": VOCAB_SIZE,
            "Data Level": args.clean,
            "Tokenizer": args.tokenizer,
            "Embedding Dimensions": EMBEDDING_DIM,
            "Embedding Layers": EMBEDDING_LAYERS,
            "Hidden Dimensions": HIDDEN_DIM,
            "Model": MODEL,
            "Comparison Type": args.comparison_type,
            "Batch Size": BATCH_SIZE,
            "Learning Optimization": args.learning_optimization,
        }
        train_stats.append(these_stats)

    return pd.DataFrame(train_stats)


def validate_model(model, val_loader, criterion):
    model.eval()
    val_preds, val_targets = [], []
    val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            text_tokens = batch["tokens"].to(DEVICE)
            text_attention_mask = batch["attention"].to(DEVICE)
            targets = batch["target"].to(DEVICE).float()

            outputs = model.forward(
                text_tokens,
                text_attention_mask,
            )

            # Compute loss for this batch
            loss = criterion(outputs.squeeze(1), targets)
            val_loss += loss.item()

            val_preds.extend((outputs > 0.5).cpu().numpy())
            val_targets.extend(targets.cpu().numpy())

    # Compute metrics
    this_accuracy_score = accuracy_score(val_targets, val_preds)
    this_precision_score = precision_score(val_targets, val_preds, zero_division=0)
    this_recall_score = recall_score(val_targets, val_preds)
    this_f1_score = f1_score(val_targets, val_preds)
    tn, fp, fn, tp = confusion_matrix(val_targets, val_preds).ravel()

    return {
        "Validation Loss": val_loss / len(val_loader),  # Average validation loss
        "Validation Accuracy Score": this_accuracy_score,
        "Validation Precision Score": this_precision_score,
        "Validation Recall Score": this_recall_score,
        "Validation F1 Score": this_f1_score,
        "Validation True Positive": tp,
        "Validation True Negative": tn,
        "Validation False Positive": fp,
        "Validation False Negative": fn,
    }


def test_model(model, test_loader):
    model.eval()
    ids = []
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            these_ids = batch["id"]
            text_tokens = batch["tokens"].to(DEVICE)
            text_attention_mask = batch["attention"].to(DEVICE)

            outputs = model.forward(
                text_tokens,
                text_attention_mask,
            )

            ids.extend(these_ids.cpu().numpy())
            predictions.extend((outputs > 0.5).cpu().numpy())

    result = pd.DataFrame({"id": ids, "target": predictions})
    result["target"] = result["target"].astype(int)
    result = result.sort_values("id")
    return result


rnn_multi_input_model = TwitterDisasterRNN(
    VOCAB_SIZE,
    use_attention=False,
    embedding_dim=EMBEDDING_DIM,
    embedding_layers=EMBEDDING_LAYERS,
    hidden_dim=HIDDEN_DIM,
    output_dim=OUTPUT_DIM,
)
rnn_multi_input_model.to(DEVICE)

train_ones_count = train_df["target"].sum()
train_zeros_count = len(train_df) - train_ones_count

train_ones_weight = len(train_df) / train_ones_count

train_zeros_weight = len(train_df) / train_zeros_count


train_stats = train_model(
    rnn_multi_input_model,
    train_combined_loader,
    val_combined_loader,
    train_zeros_weight,
    train_ones_weight,
    epochs=EPOCHS,
)


Path("./train_stats_f1").mkdir(exist_ok=True)
train_stats.to_parquet(f"./train_stats_f1/{MODEL_NAME}.parquet")

test_output_df = test_model(rnn_multi_input_model, test_combined_loader)
Path("./test_f1").mkdir(exist_ok=True)
test_output_df.to_csv(f"./test_f1/{MODEL_NAME}.csv", index=False)
