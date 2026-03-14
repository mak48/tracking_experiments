"""
Нейросетевая модель для регрессии зарплат (FCN - Fully Connected Network)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SalaryRegressorFCN(nn.Module):
    """
    Полносвязная нейронная сеть для регрессии зарплат
    Архитектура: FCN (Fully Connected Network)
    """
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.3):
        """
        Args:
            input_dim: Размерность входных признаков
            hidden_dims: Список размерностей скрытых слоев
            dropout_rate: Вероятность dropout
        """
        super(SalaryRegressorFCN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Выходной слой для регрессии (1 значение)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Инициализация весов
        self.apply(self._init_weights)
        
        logger.info(f"Создана FCN модель: {input_dim} -> {hidden_dims} -> 1")
    
    def _init_weights(self, module):
        """Инициализация весов"""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """Прямой проход"""
        return self.network(x).squeeze()


class Trainer:
    """Класс для обучения модели"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        logger.info(f"Используется устройство: {device}")
        
    def train_epoch(self, train_loader, criterion, optimizer):
        """Обучение на одной эпохе"""
        self.model.train()
        total_loss = 0
        predictions = []
        targets = []
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            
            optimizer.zero_grad()
            y_pred = self.model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(X_batch)
            predictions.extend(y_pred.detach().cpu().numpy())
            targets.extend(y_batch.cpu().numpy())
        
        avg_loss = total_loss / len(train_loader.dataset)
        return avg_loss, np.array(predictions), np.array(targets)
    
    def validate(self, val_loader, criterion):
        """Валидация"""
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch)
                
                total_loss += loss.item() * len(X_batch)
                predictions.extend(y_pred.cpu().numpy())
                targets.extend(y_batch.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader.dataset)
        return avg_loss, np.array(predictions), np.array(targets)
    
    def predict(self, X, batch_size=256):
        """Предсказание"""
        self.model.eval()
        dataset = TensorDataset(torch.FloatTensor(X))
        loader = DataLoader(dataset, batch_size=batch_size)
        
        predictions = []
        with torch.no_grad():
            for X_batch, in loader:
                X_batch = X_batch.to(self.device)
                y_pred = self.model(X_batch)
                predictions.extend(y_pred.cpu().numpy())
        
        return np.array(predictions)


def create_model(input_dim, config=None):
    """
    Фабрика для создания модели
    
    Args:
        input_dim: Размерность входных данных
        config: Словарь с конфигурацией модели
    """
    if config is None:
        config = {
            'hidden_dims': [256, 128, 64],
            'dropout_rate': 0.3,
            'learning_rate': 0.001
        }
    
    model = SalaryRegressorFCN(
        input_dim=input_dim,
        hidden_dims=config['hidden_dims'],
        dropout_rate=config['dropout_rate']
    )
    
    return model, config