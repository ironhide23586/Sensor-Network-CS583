import csv
from matplotlib import pyplot as plt
import numpy as np
import copy
from sklearn.metrics import mean_absolute_error

hum_train_means = None
hum_train_vars = None
temp_train_means = None
temp_train_vars = None

first_row = None

def compute_means(data):
    ans = dict.fromkeys(data[0].keys())
    for k in ans.keys():
        ans[k] = np.array([np.mean(d[k]) for d in data])
    return ans

def compute_vars(data):
    ans = dict.fromkeys(data[0].keys())
    for k in ans.keys():
        ans[k] = np.array([np.var(d[k]) for d in data])
    return ans

def read_data(csvObj):
    rowIdx = 0
    times = None
    cols = None
    sensor_readings = []
    for row in csvObj:
        if rowIdx == 0:
            cols = [float(r) for r in row[1:]]
            times = np.sort(np.array(list(set(cols))))
            rowIdx += 1
            sensor = dict.fromkeys(times)
            for k in sensor.keys():
                sensor[k] = []
            continue
        sensor_local = copy.deepcopy(sensor)
        row = [float(r) for r in row[1:]]
        i = 0
        for col in cols:
            sensor_local[col].append(row[i])
            i += 1
        rowIdx += 1
        sensor_readings.append(sensor_local)
    return np.array(sensor_readings)

def write_data(data, file_name):
    writer = csv.writer(open(file_name, 'wb'))
    writer.writerow(first_row)
    for i in xrange(data.shape[0]):
        row = [None] * 97
        row[0] = str(i)
        row[1:] = ['%.2E' % num for num in data[i, :]]
        writer.writerow(row)


def plot_hum_temp(t, plot='var'):
    if plot is 'var':
        sorted_args_h = np.argsort(hum_train_vars[t])
        sorted_args_t = np.argsort(temp_train_vars[t])
    elif plot is 'mean':
        sorted_args_h = np.argsort(hum_train_means[t])
        sorted_args_t = np.argsort(temp_train_means[t])
    plt.plot(range(50), hum_train_means[t][sorted_args_h])
    plt.plot(range(50), temp_train_means[t][sorted_args_t])
    plt.show()

def predict_window_inf(budget, hum_train_means, temp_train_means, hum_test_data, temp_test_data):
    times = np.sort(np.array(hum_train_means.keys()))
    times = np.append(times[1:], 0.)

    hum_test = np.ones((50, 96))
    temp_test = np.ones((50, 96))

    hum_preds = np.ones((50, 96))
    temp_preds = np.ones((50, 96))

    j = 0
    for t in times:
        hum_test[:, j] = np.array([hum_test_data[i][t][0] for i in xrange(hum_test_data.shape[0])])
        hum_test[:, j + 48] = np.array([hum_test_data[i][t][1] for i in xrange(hum_test_data.shape[0])])
        temp_test[:, j] = np.array([temp_test_data[i][t][0] for i in xrange(temp_test_data.shape[0])])
        temp_test[:, j + 48] = np.array([temp_test_data[i][t][1] for i in xrange(temp_test_data.shape[0])])
        j += 1
        
    start_hum = 0
    window_hum = None
    window_temp = None
    all_times = np.append(times, times)
    i = 0


    for t in all_times:
        end_hum = start_hum + budget
        if end_hum <= 50:
            window_hum = np.arange(start_hum, end_hum)
            start_temp = end_hum
            end_temp = start_temp + budget
            if end_temp <= 50:
                window_temp = np.arange(start_temp, end_temp)
            else:
                window_temp = np.append(np.arange(start_temp, 50), np.arange(end_temp % 50))
        else:
            start_temp = end_hum % 50
            window_hum = np.append(np.arange(start_hum, 50), np.arange(start_temp))
            window_temp = np.arange(start_temp, start_temp + budget)
        
        start_hum += budget
        start_hum %= 50



        coeff_humToTemp = hum_train_means[t] / temp_train_means[t]
        coeff_tempToHum = temp_train_means[t] / hum_train_means[t]

        hum_pred = copy.deepcopy(hum_train_means[t])
        temp_pred = copy.deepcopy(temp_train_means[t])

        hum_readings = hum_test[window_hum, i]
        temp_readings = temp_test[window_temp, i]

        hum_pred[window_hum] = copy.deepcopy(hum_readings)
        temp_pred[window_temp] = copy.deepcopy(temp_readings)

        hum_to_pred = copy.deepcopy(window_temp)
        temp_to_pred = copy.deepcopy(window_hum)

        hum_pred[hum_to_pred] = temp_pred[hum_to_pred] * coeff_humToTemp[hum_to_pred]
        temp_pred[temp_to_pred] = hum_pred[temp_to_pred] * coeff_tempToHum[temp_to_pred]

        hum_preds[:, i] = copy.deepcopy(hum_pred)
        temp_preds[:, i] = copy.deepcopy(temp_pred)
        i += 1

    hum_mean_err = mean_absolute_error(hum_test, hum_preds)
    temp_mean_err = mean_absolute_error(temp_test, temp_preds)

    return hum_preds, temp_preds, hum_mean_err, temp_mean_err


if __name__ == "__main__":
    hum_train = csv.reader(open(r'intelLabDataProcessed\intelHumidityTrain.csv', 'rb'))
    hum_test = csv.reader(open(r'intelLabDataProcessed\intelHumidityTest.csv', 'rb'))
    temp_train = csv.reader(open(r'intelLabDataProcessed\intelTemperatureTrain.csv', 'rb'))
    temp_test = csv.reader(open(r'intelLabDataProcessed\intelTemperatureTest.csv', 'rb'))

    hum_train_data = read_data(hum_train)
    hum_test_data = read_data(hum_test)
    temp_train_data = read_data(temp_train)
    temp_test_data = read_data(temp_test)

    hum_train_means = compute_means(hum_train_data)
    hum_train_vars = compute_vars(hum_train_data)
    temp_train_means = compute_means(temp_train_data)
    temp_train_vars = compute_vars(temp_train_data)

    first_row = np.ones(97)
    tmp = np.sort(np.array(hum_train_means.keys()))
    tmp = tmp[1:]
    tmp = np.append(tmp, 0.0)
    first_row[1:] = np.append(tmp, tmp)
    first_row = list(first_row)
    first_row[0] = 'sensors'

    budgets = [0, 5, 10, 20, 25]

    for budget in budgets:
        hum_preds, temp_preds, hum_mean_err, temp_mean_err = predict_window_inf(budget, hum_train_means, temp_train_means, hum_test_data, temp_test_data)
        write_data(hum_preds, str('results/humidity/w' + str(budget) + '.csv'))
        write_data(temp_preds, str('results/temperature/w' + str(budget) + '.csv'))
        lol=0
    lol=0