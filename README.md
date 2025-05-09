# Прогноза за вероятности

Автоматизирана система за предсказване на търсене, комбинираща Байесови методи, машинно обучение (Random Forest) и Монте Карло симулации.

## Как работи
1. Качете CSV файл с колони: `date` (числови дни/седмици), `sales` (продажби), `price` (цена).
2. Системата преобучава Байесов и ML модел с вашите данни.
3. Получавате:
   - Вероятност за търсене над 1000 (напр. „70%“).
   - Хистограма на прогнозираното разпределение.
4. Моделите се запазват автоматично като `bayes_posterior.pkl` и `ml_model.pkl`.

## Технически детайли
- **Байесов модел**: Използва PyMC за нормално разпределение с нормален и полунормален приор.
- **ML модел**: Random Forest Regressor с 50 дървета.
- **Монте Карло**: 5000 симулации за вероятностна прогноза.
- **Интерфейс**: Streamlit за уеб-базирано качване и визуализация.
- **Хостинг**: Streamlit Cloud.

## Демо
- Опитайте приложението: https://probability-forecast-okvckl3mhhjcx3checyybe.streamlit.app/
- Примерни данни: Изтеглете [`sales_data.csv`](./sales_data.csv) или [`new_sales_data.csv`](./new_sales_data.csv).

