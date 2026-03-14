"""
Подготовка данных для нейросетевой регрессии зарплат HH.ru
"""
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreparator:
    """Класс для подготовки данных HH.ru"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
    def load_data(self):
        """Загрузка данных"""
        logger.info(f"Загрузка данных из {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        logger.info(f"Загружено {len(self.df)} строк, {len(self.df.columns)} колонок")
        return self
    
    def extract_salary(self, salary_str):
        """Извлечение зарплаты из текста"""
        if pd.isna(salary_str):
            return np.nan
        # Убираем пробелы и берем число
        numbers = re.findall(r'\d+', str(salary_str).replace(' ', ''))
        return float(numbers[0]) if numbers else np.nan
    
    def extract_age(self, age_str):
        """Извлечение возраста"""
        if pd.isna(age_str):
            return np.nan
        match = re.search(r'(\d+)\s*(?:лет|год|года)', str(age_str))
        return int(match.group(1)) if match else np.nan
    
    def extract_experience(self, exp_str):
        """Извлечение опыта работы"""
        if pd.isna(exp_str):
            return np.nan
        # Ищем общий опыт
        match = re.search(r'Опыт работы\s+(\d+)', str(exp_str))
        if match:
            return int(match.group(1))
        # Альтернативный поиск
        match = re.search(r'(\d+)\s*(?:лет|год|года)', str(exp_str))
        return int(match.group(1)) if match else np.nan
    
    def extract_city(self, city_str):
        """Извлечение города"""
        if pd.isna(city_str):
            return 'unknown'
        return str(city_str).split(',')[0].strip()
    
    def extract_position_features(self, position_str):
        """Извлечение признаков из названия должности"""
        if pd.isna(position_str):
            return {}
        
        text = str(position_str).lower()
        features = {}
        
        # Уровневые ключевые слова
        features['has_junior'] = 1 if any(word in text for word in ['junior', 'джуниор', 'стажер']) else 0
        features['has_middle'] = 1 if any(word in text for word in ['middle', 'мидл']) else 0
        features['has_senior'] = 1 if any(word in text for word in ['senior', 'сеньор', 'ведущий']) else 0
        features['has_lead'] = 1 if any(word in text for word in ['lead', 'тимлид', 'руководитель']) else 0
        
        # Специализации
        specs = {
            'backend': ['backend', 'бэкенд', 'бэкэнд'],
            'frontend': ['frontend', 'фронтенд', 'front-end'],
            'fullstack': ['fullstack', 'фуллстек', 'full-stack'],
            'devops': ['devops', 'девопс'],
            'data': ['data', 'дата', 'аналитик'],
            'mobile': ['mobile', 'мобильный'],
            'qa': ['qa', 'тестировщик', 'тестирование']
        }
        
        for spec, keywords in specs.items():
            features[f'spec_{spec}'] = 1 if any(k in text for k in keywords) else 0
        
        return features
    
    def extract_skills(self, text):
        """Извлечение навыков из текста резюме"""
        if pd.isna(text):
            return []
        
        text = str(text).lower()
        
        # Список популярных IT-навыков
        skills_list = [
            'python', 'java', 'javascript', 'js', 'c#', 'c++', 'php', 'ruby', 'go',
            'sql', 'mysql', 'postgresql', 'mongodb', 'redis',
            'django', 'flask', 'spring', 'laravel', 'rails',
            'react', 'vue', 'angular', 'node', 'jquery',
            'docker', 'kubernetes', 'aws', 'azure', 'gcp',
            'git', 'linux', 'bash', 'nginx', 'apache',
            'tensorflow', 'pytorch', 'scikit-learn', 'pandas',
            'html', 'css', 'bootstrap'
        ]
        
        skills_found = []
        for skill in skills_list:
            if skill in text:
                skills_found.append(skill)
                # Создаем бинарный признак для каждого навыка
                self.skill_features[f'skill_{skill}'] = 1
        
        return skills_found
    
    def prepare_features(self):
        """Подготовка признаков для модели"""
        logger.info("Подготовка признаков...")
        
        # Словарь для бинарных признаков навыков
        self.skill_features = {}
        
        # Целевая переменная - зарплата
        logger.info("Извлечение зарплаты...")
        self.df['salary'] = self.df['ЗП'].apply(self.extract_salary)
        
        # Извлечение признаков
        logger.info("Извлечение возраста...")
        self.df['age'] = self.df['Пол, возраст'].apply(self.extract_age)
        
        logger.info("Извлечение опыта...")
        self.df['experience'] = self.df['Опыт (двойное нажатие для полной версии)'].apply(self.extract_experience)
        
        logger.info("Извлечение города...")
        self.df['city'] = self.df['Город'].apply(self.extract_city)
        
        # Признаки из должности
        logger.info("Извлечение признаков из должности...")
        position_features = self.df['Ищет работу на должность:'].apply(self.extract_position_features)
        position_df = pd.json_normalize(position_features)
        self.df = pd.concat([self.df, position_df], axis=1)
        
        # Удаляем строки с пропусками в целевой переменной
        initial_len = len(self.df)
        self.df = self.df.dropna(subset=['salary', 'age', 'experience'])
        logger.info(f"Удалено строк с пропусками: {initial_len - len(self.df)}")
        
        # Кодирование городов
        logger.info("Кодирование городов...")
        top_cities = self.df['city'].value_counts().head(20).index
        for city in top_cities:
            self.df[f'city_{city}'] = (self.df['city'] == city).astype(int)
        
        # Сбор всех признаков
        feature_cols = ['age', 'experience']
        feature_cols.extend([col for col in self.df.columns if col.startswith('has_')])
        feature_cols.extend([col for col in self.df.columns if col.startswith('spec_')])
        feature_cols.extend([col for col in self.df.columns if col.startswith('city_')])
        
        self.feature_names = feature_cols
        X = self.df[feature_cols].fillna(0).values
        y = self.df['salary'].values
        
        logger.info(f"Подготовлено признаков: {X.shape}")
        logger.info(f"Целевая переменная: {y.shape}")
        logger.info(f"Диапазон зарплат: {y.min():.0f} - {y.max():.0f}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, val_size=0.1):
        """Разделение данных на train/val/test"""
        logger.info("Разделение данных...")
        
        # Сначала разделяем на train+val и test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Затем разделяем train+val на train и val
        val_relative_size = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_relative_size, random_state=42
        )
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Масштабирование
        logger.info("Масштабирование признаков...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        return (X_train_scaled, y_train), (X_val_scaled, y_val), (X_test_scaled, y_test)
    
    def save_preprocessor(self, path="models/preprocessor.pkl"):
        """Сохранение препроцессора"""
        Path("models").mkdir(exist_ok=True)
        preprocessor = {
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'label_encoders': self.label_encoders
        }
        with open(path, 'wb') as f:
            pickle.dump(preprocessor, f)
        logger.info(f"Препроцессор сохранен в {path}")


def prepare_data_for_training(data_path="hh.csv"):
    """Основная функция подготовки данных"""
    preparator = DataPreparator(data_path)
    preparator.load_data()
    X, y = preparator.prepare_features()
    splits = preparator.split_data(X, y)
    preparator.save_preprocessor()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = splits
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), preparator.feature_names