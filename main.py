import warnings
from datetime import datetime
from calendar import isleap
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

wines_data = pd.read_csv('wine_Austral.dat', delimiter='\t')

# Преобразование строковых дат в объекты datetime
months_days = {'1': 31, '2': 28, '3': 31, '4': 30,
               '5': 31, '6': 30, '7': 31, '8': 31, '9': 30,
               '10': 31, '11': 30, '12': 31}

wines_data['month'] = wines_data['month_'].astype(str) + ' ' + wines_data['year_'].astype(str)
months_list = []
for m_date in wines_data['month']:
    month_values = m_date.split(' ')
    if isleap(int(month_values[1])) and month_values[0] == '2':
        last_day = 29
    else:
        last_day = months_days[month_values[0]]
    month_date = datetime.strptime(str(last_day) + ' ' + m_date, '%d %m %Y')
    months_list.append(month_date)

del wines_data['month']
del wines_data['month_']
del wines_data['year_']

wines_data.insert(0, 'month', months_list)

# Выборка столбцов из набора данных
new_wine_data = pd.concat([wines_data['month'], wines_data['fort']], axis=1)
new_wine_data.set_index('month', inplace=True)
new_wine_data.index = pd.to_datetime(new_wine_data.index)

new_wine_data = new_wine_data.asfreq('m')

decompose = seasonal_decompose(new_wine_data)

f1 = plt.figure()
f2 = plt.figure()
f3 = plt.figure()
# График тренда
ax1 = f1.add_subplot(111)
ax1.plot(decompose.trend, color='blue')
ax1.set_title('Тренд')
ax1.set_xlabel('Год')
ax1.set_ylabel('Литры (в тыс.)')
ax1.grid(True)
# График сезонности
ax2 = f2.add_subplot(111)
ax2.plot(decompose.seasonal, color='blue')
ax2.set_title('Сезонность')
ax2.set_xlabel('Год')
ax2.set_ylabel('Литры (в тыс.)')
ax2.grid(True)

# Прогнозирование - обучение модели sarimax
learn_dataset = new_wine_data[:]

warnings.simplefilter(action='ignore', category=Warning)
model = SARIMAX(learn_dataset, order=(3, 0, 0), seasonal_order=(0, 1, 0, 12))

result = model.fit()

# Предсказываем поведение ряда на последующие 8 месяцев
start = len(learn_dataset)
end = len(learn_dataset) + 8

predictions = result.predict(start, end)
print('\n' + 'Спрогнозированные на 8 месяцев значения:')
print(predictions)

# Исходный ряд и значения прогноза
ax3 = f3.add_subplot(111)
ax3.plot(new_wine_data, color='blue')
ax3.plot(predictions, color='red')
ax3.set_title('Потребление крепленого вина')
ax3.set_xlabel('Год')
ax3.set_ylabel('Литры (в тыс.)')
ax3.grid(True)

plt.show()
