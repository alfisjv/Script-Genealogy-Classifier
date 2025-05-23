# train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import learning_rate, patience, min_delta, epochs
from utils import logger

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Logger CSV setup
os.makedirs("logs", exist_ok=True)
csv_log_path = "logs/training_log.csv"
log_columns = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "test_loss", "test_acc", "f1_micro", "f1_macro"]
log_df = pd.DataFrame(columns=log_columns)

# Optional: fallback global placeholders (legacy support)
val_loader_global = None
test_loader_global = None

def set_eval_loaders(val_dl, test_dl):
    global val_loader_global, test_loader_global
    val_loader_global = val_dl
    test_loader_global = test_dl

def evaluate(model, dataloader, loss_fn):
    model.eval()
    correct, total_loss = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return correct / len(dataloader.dataset), total_loss / len(dataloader.dataset), all_preds, all_labels

def train_model(model, num_epochs, train_loader, loss_fn, optimizer, val_loader=None, test_loader=None):
    global val_loader_global, test_loader_global

    # Fall back to globals if not passed explicitly
    if val_loader is None or test_loader is None:
        val_loader = val_loader_global
        test_loader = test_loader_global

    if val_loader is None or test_loader is None:
        print("[Warning] Validation/Test loaders not provided. Skipping validation and test evaluation.")

    os.makedirs("checkpoints", exist_ok=True)

    best_val_loss = float('inf')
    early_stop_counter = 0

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct = 0, 0
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Training...")

        for images, labels in tqdm(train_loader, leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()

        train_acc = train_correct / len(train_loader.dataset)

        # Validate if loaders are available
        if val_loader:
            val_acc, val_loss, val_preds, val_labels = evaluate(model, val_loader, loss_fn)
        else:
            val_acc, val_loss = None, None

        if test_loader:
            test_acc, test_loss, test_preds, test_labels = evaluate(model, test_loader, loss_fn)
            f1_micro = f1_score(test_labels, test_preds, average='micro')
            f1_macro = f1_score(test_labels, test_preds, average='macro')
        else:
            test_acc, test_loss, f1_micro, f1_macro = None, None, None, None

        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        if val_loss is not None:
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        if test_loss is not None:
            logger.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
            logger.info(f"F1 Micro: {f1_micro:.4f}, F1 Macro: {f1_macro:.4f}")

        # Save best model based on validation loss
        if val_loss is not None and (best_val_loss - val_loss > min_delta):
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), "checkpoints/final_weights.pth")
            logger.info("Validation loss improved. Model saved!")
        else:
            early_stop_counter += 1
            logger.warning(f"No improvement for {early_stop_counter} epoch(s).")
            if early_stop_counter >= patience:
                logger.error("Early stopping triggered!")
                break

        # Save log
        log_df.loc[len(log_df)] = [epoch+1, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc, f1_micro, f1_macro]
        log_df.to_csv(csv_log_path, index=False)

        # Save confusion matrix if test_loader available
        if test_loader:
            cm = classification_report(test_labels, test_preds, output_dict=True)
            plt.figure(figsize=(6, 5))
            sns.heatmap(pd.DataFrame(cm).iloc[:-1, :-1], annot=True)
            plt.title(f"Confusion Matrix - Epoch {epoch+1}")
            plt.savefig(f"logs/conf_matrix_epoch{epoch+1}.png")
            plt.close()

    return model
