"""
Обучение модели с MLflow трекингом
"""
import numpy as np
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import logging
import sys
from pathlib import Path
import json

# Добавляем путь к проекту
sys.path.append(str(Path(__file__).parent.parent))

from src.data_preparation import prepare_data_for_training
from src.neural_network import create_model, Trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Класс для трекинга экспериментов в MLflow"""
    
    def __init__(self, experiment_name, tracking_uri="http://kamnsv.com:55000"):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        
        # Настройка MLflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        
        logger.info(f"MLflow tracking URI: {tracking_uri}")
        logger.info(f"Эксперимент: {experiment_name}")
    
    def log_params(self, params):
        """Логирование параметров"""
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    def log_metrics(self, metrics, step=None):
        """Логирование метрик"""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_artifact(self, local_path):
        """Логирование артефакта"""
        mlflow.log_artifact(local_path)
    
    def log_model(self, model, model_name):
        """Логирование модели"""
        mlflow.pytorch.log_model(model, model_name)


def train_model(config):
    """Основная функция обучения"""
    
    # Подготовка данных
    logger.info("=" * 60)
    logger.info("ПОДГОТОВКА ДАННЫХ")
    logger.info("=" * 60)
    
    (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_names = prepare_data_for_training()
    
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Создание даталоадеров
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
    
    # Создание модели
    logger.info("\n" + "=" * 60)
    logger.info("СОЗДАНИЕ МОДЕЛИ")
    logger.info("=" * 60)
    
    input_dim = X_train.shape[1]
    model, model_config = create_model(input_dim, config['model'])
    
    # Параметры обучения
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    trainer = Trainer(model)
    
    # Трекинг в MLflow
    tracker = ExperimentTracker(
        experiment_name=config['experiment_name'],
        tracking_uri=config['tracking_uri']
    )
    
    with mlflow.start_run(run_name=config['run_name']):
        # Логирование параметров
        params = {
            'input_dim': input_dim,
            'hidden_dims': config['model']['hidden_dims'],
            'dropout_rate': config['model']['dropout_rate'],
            'batch_size': config['batch_size'],
            'learning_rate': config['learning_rate'],
            'n_epochs': config['n_epochs'],
            'optimizer': 'Adam',
            'criterion': 'MSE',
            'feature_count': len(feature_names)
        }
        tracker.log_params(params)
        
        # Обучение
        logger.info("\n" + "=" * 60)
        logger.info("ОБУЧЕНИЕ МОДЕЛИ")
        logger.info("=" * 60)
        
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(config['n_epochs']):
            # Обучение
            train_loss, train_pred, train_true = trainer.train_epoch(train_loader, criterion, optimizer)
            train_r2 = r2_score(train_true, train_pred)
            train_mae = mean_absolute_error(train_true, train_pred)
            
            # Валидация
            val_loss, val_pred, val_true = trainer.validate(val_loader, criterion)
            val_r2 = r2_score(val_true, val_pred)
            val_mae = mean_absolute_error(val_true, val_pred)
            
            scheduler.step(val_loss)
            
            # Логирование метрик
            metrics = {
                'train_loss': train_loss,
                'train_r2': train_r2,
                'train_mae': train_mae,
                'val_loss': val_loss,
                'val_r2': val_r2,
                'val_mae': val_mae,
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            tracker.log_metrics(metrics, step=epoch)
            
            # Логирование каждые 10 эпох
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{config['n_epochs']}: "
                           f"Train Loss: {train_loss:.2f}, Train R2: {train_r2:.4f}, "
                           f"Val Loss: {val_loss:.2f}, Val R2: {val_r2:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                # Сохраняем лучшую модель
                torch.save(model.state_dict(), 'models/best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= config['early_stopping_patience']:
                    logger.info(f"Early stopping на эпохе {epoch+1}")
                    break
        
        # Загрузка лучшей модели
        model.load_state_dict(torch.load('models/best_model.pth'))
        trainer.model = model
        
        # Оценка на тестовой выборке
        logger.info("\n" + "=" * 60)
        logger.info("ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ")
        logger.info("=" * 60)
        
        test_loss, test_pred, test_true = trainer.validate(test_loader, criterion)
        test_r2 = r2_score(test_true, test_pred)
        test_mae = mean_absolute_error(test_true, test_pred)
        test_rmse = np.sqrt(mean_squared_error(test_true, test_pred))
        
        # Логирование тестовых метрик
        test_metrics = {
            'test_loss': test_loss,
            'r2_score_test': test_r2,  # Важно: именно такое название для задания
            'test_mae': test_mae,
            'test_rmse': test_rmse
        }
        tracker.log_metrics(test_metrics)
        
        logger.info(f"Test R2: {test_r2:.4f}")
        logger.info(f"Test MAE: {test_mae:.2f} руб.")
        logger.info(f"Test RMSE: {test_rmse:.2f} руб.")
        
        # Сохранение модели в MLflow
        model_name = config['model_name']
        tracker.log_model(model, model_name)
        
        # Сохранение дополнительных артефактов
        with open('models/feature_names.json', 'w') as f:
            json.dump(feature_names, f)
        tracker.log_artifact('models/feature_names.json')
        
        # Получаем run_id для вывода
        run_id = mlflow.active_run().info.run_id
        logger.info(f"\n✅ Эксперимент завершен!")
        logger.info(f"   Run ID: {run_id}")
        logger.info(f"   Модель: {model_name}")
        logger.info(f"   R2 на тесте: {test_r2:.4f}")
        
        return {
            'run_id': run_id,
            'test_r2': test_r2,
            'model_name': model_name
        }


def main():
    """Основная функция"""
    
    # Конфигурация эксперимента
    config = {
        # MLflow настройки
        'experiment_name': 'LIne Regression HH',
        'tracking_uri': 'http://kamnsv.com:55000',
        'model_name': 'makoveeva_anastasia_fcn',  # FCN - полносвязная сеть
        
        # Параметры модели
        'model': {
            'hidden_dims': [256, 128, 64, 32],
            'dropout_rate': 0.3,
        },
        
        # Параметры обучения
        'batch_size': 64,
        'learning_rate': 0.001,
        'n_epochs': 100,
        'early_stopping_patience': 10,
        
        # Название запуска
        'run_name': 'fcn_experiment_v1'
    }
    
    # Обучение
    results = train_model(config)
    
    print("\n" + "=" * 60)
    print("ИТОГИ ЭКСПЕРИМЕНТА")
    print("=" * 60)
    print(f"✅ RUN_ID: {results['run_id']}")
    print(f"✅ Модель: {results['model_name']}")
    print(f"✅ R2 на тесте: {results['test_r2']:.4f}")
    print(f"\n👉 Ссылка на MLflow: {config['tracking_uri']}")
    print("=" * 60)


if __name__ == "__main__":
    main()