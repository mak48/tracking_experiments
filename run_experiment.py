"""
Запуск эксперимента с разными конфигурациями
"""
import subprocess
import sys
from pathlib import Path

# Создаем необходимые директории
Path("models").mkdir(exist_ok=True)
Path("mlruns").mkdir(exist_ok=True)

# Запуск обучения
print("=" * 60)
print("ЗАПУСК ЭКСПЕРИМЕНТА: РЕГРЕССИЯ ЗАРПЛАТ HH.RU")
print("=" * 60)

# Запускаем обучение
subprocess.run([sys.executable, "src/train.py"])

print("\n✅ Эксперимент завершен!")
print("👉 Проверьте результаты в MLflow: http://kamnsv.com:55000")