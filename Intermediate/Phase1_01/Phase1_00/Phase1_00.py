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
global_times = None

hum_test = None
temp_test = None

def compute_means(data):
    """Computes the means of the input observations over the days."""
    ans = dict.fromkeys(data[0].keys())
    for k in ans.keys():
        ans[k] = np.array([np.mean(d[k]) for d in data])
    return ans

def compute_vars(data):
    """Computes the variances of the input observations over the days."""
    ans = dict.fromkeys(data[0].keys())
    for k in ans.keys():
        ans[k] = np.array([np.var(d[k]) for d in data])
    return ans

def read_data(csvObj):
    """Method to read and parse data from csv file."""
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
    """Method to write formatted results to csv files."""
    writer = csv.writer(open(file_name, 'wb'))
    writer.writerow(first_row)
    for i in xrange(data.shape[0]):
        row = [None] * 97
        row[0] = str(i)
        row[1:] = ['%.2E' % num for num in data[i, :]]
        writer.writerow(row)

def makeMatrix(dictData):
    """Makes a 2D numpy matrix out of input data structure which contains test data in the form of list of dictionaries."""
    res = np.ones((50, 96))
    j = 0
    for t in global_times[:48]:
        res[:, j] = np.array([dictData[i][t][0] for i in xrange(dictData.shape[0])])
        res[:, j + 48] = np.array([dictData[i][t][1] for i in xrange(dictData.shape[0])])
        j += 1
    return res

def makePreds(window_hum, window_temp, hum_train_means, temp_train_means, timeIdx, timeVal):
    """Computes predictions of different sensors at a given time, given the indices of those sensors for which
    the reading has been obtained (window_hum and window_temp)."""

    #Firstly ratios between the temperature and humidity readings are computed which are subsequently used to make predictions
    coeff_humToTemp = hum_train_means[timeVal] / temp_train_means[timeVal] #Used to predict humidities
    coeff_tempToHum = temp_train_means[timeVal] / hum_train_means[timeVal] #Used to predict temperatures

    #The correlations between temperature and humidity above are used to make better predictions of each other.

    hum_pred = copy.deepcopy(hum_train_means[timeVal])
    temp_pred = copy.deepcopy(temp_train_means[timeVal])

    if window_hum.shape[0] > 0:
        hum_readings = hum_test[window_hum, timeIdx]
        temp_readings = temp_test[window_temp, timeIdx]

        hum_pred[window_hum] = copy.deepcopy(hum_readings)
        temp_pred[window_temp] = copy.deepcopy(temp_readings)

        set_wh = set(window_hum)
        set_wt = set(window_temp)

        hum_to_pred = np.array(list(set_wt - set_wh))
        temp_to_pred = np.array(list(set_wh - set_wt))

        if hum_to_pred.shape[0] > 0:
            hum_pred[hum_to_pred] = temp_pred[hum_to_pred] * coeff_humToTemp[hum_to_pred]
        if temp_to_pred.shape[0] > 0:
            temp_pred[temp_to_pred] = hum_pred[temp_to_pred] * coeff_tempToHum[temp_to_pred]

    return hum_pred, temp_pred

def predict_window_inf(budget, hum_train_means, temp_train_means):
    """Method to make predictions based on windowed active inference."""        
    start_hum = 0
    window_hum = None
    window_temp = None
    i = 0

    hum_preds = np.ones((50, 96))
    temp_preds = np.ones((50, 96))

    for t in global_times:
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

        hum_pred, temp_pred = makePreds(window_hum, window_temp, hum_train_means, temp_train_means, i, t)

        hum_preds[:, i] = copy.deepcopy(hum_pred)
        temp_preds[:, i] = copy.deepcopy(temp_pred)
        
        i += 1

    hum_mean_err = mean_absolute_error(hum_test, hum_preds)
    temp_mean_err = mean_absolute_error(temp_test, temp_preds)

    return hum_preds, temp_preds, hum_mean_err, temp_mean_err

def predict_variance_inf(budget, hum_train_means, temp_train_means, hum_train_var, temp_train_vars):
    """Method to make predictions based on max-variance active inference."""           
    start_hum = 0
    window_hum = None
    window_temp = None
    i = 0

    hum_preds = np.ones((50, 96))
    temp_preds = np.ones((50, 96))

    for t in global_times:
        if budget > 0:
            window_hum = np.argpartition(hum_train_vars[t], -budget)[-budget:]
            window_temp = np.argpartition(temp_train_vars[t], -budget)[-budget:]
        else:
            window_hum = np.array([])
            window_temp = np.array([])

        hum_pred, temp_pred = makePreds(window_hum, window_temp, hum_train_means, temp_train_means, i, t)

        hum_preds[:, i] = copy.deepcopy(hum_pred)
        temp_preds[:, i] = copy.deepcopy(temp_pred)
        
        i += 1

    hum_mean_err = mean_absolute_error(hum_test, hum_preds)
    temp_mean_err = mean_absolute_error(temp_test, temp_preds)

    return hum_preds, temp_preds, hum_mean_err, temp_mean_err


if __name__ == "__main__":
    """Creating CSV reader objects."""
    hum_train = csv.reader(open(r'intelLabDataProcessed\intelHumidityTrain.csv', 'rb'))
    hum_test = csv.reader(open(r'intelLabDataProcessed\intelHumidityTest.csv', 'rb'))
    temp_train = csv.reader(open(r'intelLabDataProcessed\intelTemperatureTrain.csv', 'rb'))
    temp_test = csv.reader(open(r'intelLabDataProcessed\intelTemperatureTest.csv', 'rb'))

    """Using the CSV reader objects to read and parse data into variables."""
    hum_train_data = read_data(hum_train)
    hum_test_data = read_data(hum_test)
    temp_train_data = read_data(temp_train)
    temp_test_data = read_data(temp_test)

    """Computing prediction models by calculating means and variances of observations."""
    hum_train_means = compute_means(hum_train_data)
    hum_train_vars = compute_vars(hum_train_data)
    temp_train_means = compute_means(temp_train_data)
    temp_train_vars = compute_vars(temp_train_data)

    """Initializing the first row to be written on to the CSV files."""
    first_row = np.ones(97)
    tmp = np.sort(np.array(hum_train_means.keys()))
    tmp = tmp[1:]
    tmp = np.append(tmp, 0.0)
    first_row[1:] = np.append(tmp, tmp)
    global_times = copy.deepcopy(first_row[1:])
    first_row = list(first_row)
    first_row[0] = 'sensors'

    hum_test = makeMatrix(hum_test_data)
    temp_test = makeMatrix(temp_test_data)

    budgets = [0, 5, 10, 20, 25]

    hum_errors_win = []
    temp_errors_win = []
    hum_errors_var = []
    temp_errors_var = []

    """Computing predictions."""
    for budget in budgets:
        hum_preds, temp_preds, hum_mean_err, temp_mean_err = predict_window_inf(budget, hum_train_means, temp_train_means)
        write_data(hum_preds, str('results/humidity/w' + str(budget) + '.csv'))
        write_data(temp_preds, str('results/temperature/w' + str(budget) + '.csv'))
        hum_errors_win.append(hum_mean_err)
        temp_errors_win.append(temp_mean_err)

        hum_preds, temp_preds, hum_mean_err, temp_mean_err = predict_variance_inf(budget, hum_train_means, temp_train_means, hum_train_vars, temp_train_vars)
        write_data(hum_preds, str('results/humidity/v' + str(budget) + '.csv'))
        write_data(temp_preds, str('results/temperature/v' + str(budget) + '.csv'))
        hum_errors_var.append(hum_mean_err)
        temp_errors_var.append(temp_mean_err)

    """Plotting Histograms."""

    plt.xticks(range(len(budgets)), budgets)
    plt.bar(range(len(budgets)), hum_errors_win, align='center', alpha=0.4)
    plt.xlabel('$BUDGETS$')
    plt.ylabel('$Mean Absolute Error$')
    plt.title("Humidity Error vs Budgets (Windowed Active Inference)")
    plt.show()

    plt.xticks(range(len(budgets)), budgets)
    plt.bar(range(len(budgets)), temp_errors_win, align='center', alpha=0.4)
    plt.xlabel('$BUDGETS$')
    plt.ylabel('$Mean Absolute Error$')
    plt.title("Temperature Error vs Budgets (Windowed Active Inference)")
    plt.show()

    plt.xticks(range(len(budgets)), budgets)
    plt.bar(range(len(budgets)), hum_errors_var, align='center', alpha=0.4)
    plt.xlabel('$BUDGETS$')
    plt.ylabel('$Mean Absolute Error$')
    plt.title("Humidity Error vs Budgets (Variance Active Inference)")
    plt.show()

    plt.xticks(range(len(budgets)), budgets)
    plt.bar(range(len(budgets)), temp_errors_var, align='center', alpha=0.4)
    plt.xlabel('$BUDGETS$')
    plt.ylabel('$Mean Absolute Error$')
    plt.title("Temperature Error vs Budgets (Variance Active Inference)")
    plt.show()