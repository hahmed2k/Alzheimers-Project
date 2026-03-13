import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_and_preprocess_data(file_path='data/alzheimers_disease_data.csv'):
    df = pd.read_csv(file_path)
    df = df.drop(['PatientID', 'DoctorInCharge'], axis=1)
    
    # Feature engineering
    df['BP_Interaction'] = abs(df['SystolicBP'] - df['DiastolicBP'])
    df['Cholesterol_Ratio'] = df['CholesterolLDL'] / df['CholesterolHDL']
    df['TG_HDL_Index'] = df['CholesterolTriglycerides'] / df['CholesterolHDL']
    
    X = df.drop('Diagnosis', axis=1)
    y = df['Diagnosis']
    
    cat_cols = ['Ethnicity', 'EducationLevel']
    num_cols = [col for col in X.columns if col not in cat_cols]
    
    scaler = StandardScaler()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test, random_state=42)
    
    X_train_num = scaler.fit_transform(X_train[num_cols].values)
    X_val_num = scaler.transform(X_val[num_cols].values)
    X_test_num = scaler.transform(X_test[num_cols].values)
    
    X_train_cat = X_train[cat_cols].values
    X_val_cat = X_val[cat_cols].values
    X_test_cat = X_test[cat_cols].values
    
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    
    # Tensors
    X_train_num_t = torch.tensor(X_train_num, dtype=torch.float32)
    X_train_cat_t = torch.tensor(X_train_cat, dtype=torch.long)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    
    X_val_num_t = torch.tensor(X_val_num, dtype=torch.float32)
    X_val_cat_t = torch.tensor(X_val_cat, dtype=torch.long)
    y_val_t = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
    
    X_test_num_t = torch.tensor(X_test_num, dtype=torch.float32)
    X_test_cat_t = torch.tensor(X_test_cat, dtype=torch.long)
    y_test_t = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
    
    # DataLoaders
    batch_size = 128
    train_loader = DataLoader(TensorDataset(X_train_num_t, X_train_cat_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_num_t, X_val_cat_t, y_val_t), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_num_t, X_test_cat_t, y_test_t), batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, class_weights, len(num_cols), [X[col].nunique() for col in cat_cols]

if __name__ == "__main__":
    # Test run
    load_and_preprocess_data()