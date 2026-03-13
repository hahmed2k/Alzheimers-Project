import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, precision_score, recall_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, train_loader, val_loader, class_weights, epochs=50, lr=0.001, patience=15):
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for num_b, cat_b, labels in train_loader:
            num_b, cat_b, labels = num_b.to(device), cat_b.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(num_b, cat_b)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        val_loss, _ = evaluate_model(model, val_loader, criterion)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    
    model.load_state_dict(torch.load('best_model.pth'))
    return train_losses, val_losses

def evaluate_model(model, loader, criterion=None):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for num_b, cat_b, labels in loader:
            num_b, cat_b, labels = num_b.to(device), cat_b.to(device), labels.to(device)
            outputs = model(num_b, cat_b)
            if criterion:
                total_loss += criterion(outputs, labels).item()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())
    avg_loss = total_loss / len(loader) if criterion else None
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    specificity = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
    return avg_loss, {
        'Accuracy': acc, 'F1-Score': f1, 'AUC-ROC': auc,
        'Precision': precision, 'Recall': recall, 'Specificity': specificity,
        'Confusion Matrix': cm
    }
