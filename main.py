import torch
from preprocess import load_and_preprocess_data
from models import AlzheimerMLP, TabTransformer
from utils import train_model, evaluate_model
import matplotlib.pyplot as plt
import seaborn as sns

train_loader, val_loader, test_loader, class_weights, num_numerical, cat_sizes = load_and_preprocess_data()

# Train MLP
model_mlp = AlzheimerMLP(num_numerical, cat_sizes)
model_mlp.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
train_losses_mlp, val_losses_mlp = train_model(model_mlp, train_loader, val_loader, class_weights)

_, metrics_mlp = evaluate_model(model_mlp, test_loader)
print("MLP Test Metrics:", metrics_mlp)

plt.plot(train_losses_mlp, label='Train')
plt.plot(val_losses_mlp, label='Val')
plt.title('MLP Training Curves')
plt.legend()
plt.show()

sns.heatmap(metrics_mlp['Confusion Matrix'], annot=True, fmt='d')
plt.title('MLP Confusion Matrix')
plt.show()

# Train TabTransformer
model_tab = TabTransformer(num_numerical, cat_sizes)
model_tab.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
train_losses_tab, val_losses_tab = train_model(model_tab, train_loader, val_loader, class_weights)

_, metrics_tab = evaluate_model(model_tab, test_loader)
print("TabTransformer Test Metrics:", metrics_tab)

plt.plot(train_losses_tab, label='Train')
plt.plot(val_losses_tab, label='Val')
plt.title('TabTransformer Training Curves')
plt.legend()
plt.show()

sns.heatmap(metrics_tab['Confusion Matrix'], annot=True, fmt='d')
plt.title('TabTransformer Confusion Matrix')
plt.show()