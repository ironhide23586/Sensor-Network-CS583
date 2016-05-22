import csv
from matplotlib import pyplot as plt
import numpy as np
import copy
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
import collections
from sklearn.linear_model import Lasso

hum_train_means = None
hum_train_vars = None
temp_train_means = None
temp_train_vars = None

first_row = None
global_times = None

hum_test = None
temp_test = None


def compute_means_daylvl(data):
    """Computes the means of the input observations over the days."""
    ans = dict.fromkeys(data[0].keys())
    for k in ans.keys():
        ans[k] = np.array([np.mean(d[k]) for d in data])
    return ans

def compute_means_hrlvl(data):
    """Computes the means of the input observations of all sensors over all the observation times (hour level)."""
    return np.array([np.mean([np.mean(elem) for elem in data_elem.values()]) for data_elem in data])

def compute_vars_daylvl(data):
    """Computes the variances of the input observations over the days."""
    ans = dict.fromkeys(data[0].keys())
    for k in ans.keys():
        ans[k] = np.array([np.var(d[k]) for d in data])
    return ans

def compute_vars_hrlvl(data):
    """Computes the variances of the input observations of all sensors over all the observation times (hour level)."""
    ret = []
    for data_elem in data:
        tmp = []
        [[tmp.append(elem) for elem in v] for v in data_elem.values()]
        ret.append(np.var(tmp))
    return np.array(ret)

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

def makePreds(window_, preds, clfs, timeIdx, dType):
    """Computes predictions of different sensors at a given time, given the indices of those sensors for which
    the reading has been obtained."""
    #poly = PolynomialFeatures(1)
    #x = poly.fit_transform(preds.reshape(1, -1))[0]
    preds = np.array([clf.predict(preds.reshape(1, -1))[0] for clf in clfs])

    if dType == 'humidity':
        preds[window_] = copy.deepcopy(hum_test[window_, timeIdx])
    elif dType == 'temperature':
        preds[window_] = copy.deepcopy(temp_test[window_, timeIdx])

    return preds

def makePreds_phase2(window_, preds, clfs, timeIdx, dType):
    """Computes predictions of different sensors at a given time, given the indices of those sensors for which
    the reading has been obtained."""
    #poly = PolynomialFeatures(1)
    #x = poly.fit_transform(preds.reshape(1, -1))[0]
    preds = np.array([clfs[i].predict(preds[i])[0] for i in xrange(clfs.shape[0])])

    if dType == 'humidity':
        preds[window_] = copy.deepcopy(hum_test[window_, timeIdx])
    elif dType == 'temperature':
        preds[window_] = copy.deepcopy(temp_test[window_, timeIdx])

    return preds


def predict_window_inf_hrlvl(budget, train_means, betas, dType):
    """Method to make predictions based on windowed active inference (Hour level model)."""        
    start_ = 0
    window_ = None
    i = 1

    final_preds = np.ones((50, 96))

    end_ = start_ + budget
    if end_ <= 50:
        window_ = np.arange(start_, end_)
    else:
        window_ = np.append(np.arange(start_, 50), np.arange(end_ % 50))

    start_ += budget
    start_ %= 50

    preds = copy.deepcopy(train_means)

    if dType == 'humidity':
        preds[window_] = copy.deepcopy(hum_test[window_, 0])
    elif dType == 'temperature':
        preds[window_] = copy.deepcopy(temp_test[window_, 0])

    final_preds[:, 0] = copy.deepcopy(preds)

    for t in global_times[1:]:
        end_ = start_ + budget
        if end_ <= 50:
            window_ = np.arange(start_, end_)
        else:
            window_ = np.append(np.arange(start_, 50), np.arange(end_ % 50))

        start_ += budget
        start_ %= 50

        preds = makePreds(window_, preds, betas, i, dType)
        #print mean_absolute_error(hum_test[:, i], preds), window_
        final_preds[:, i] = copy.deepcopy(preds)

        i += 1

    if dType == 'humidity':
        final_mean_err = mean_absolute_error(hum_test, final_preds)
    elif dType == 'temperature':
        final_mean_err = mean_absolute_error(temp_test, final_preds)

    return final_preds, final_mean_err;


def predict_window_inf_daylvl(budget, train_means, betas, dType):
    start_ = 0
    window_ = None
    i = 1

    final_preds = np.ones((50, 96))

    end_ = start_ + budget
    if end_ <= 50:
        window_ = np.arange(start_, end_)
    else:
        window_ = np.append(np.arange(start_, 50), np.arange(end_ % 50))

    start_ += budget
    start_ %= 50

    preds = copy.deepcopy(train_means)

    if dType == 'humidity':
        preds[window_] = copy.deepcopy(hum_test[window_, 0])
    elif dType == 'temperature':
        preds[window_] = copy.deepcopy(temp_test[window_, 0])

    final_preds[:, 0] = copy.deepcopy(preds)

    for t in global_times[1:]:
        end_ = start_ + budget
        if end_ <= 50:
            window_ = np.arange(start_, end_)
        else:
            window_ = np.append(np.arange(start_, 50), np.arange(end_ % 50))

        start_ += budget
        start_ %= 50

        preds = makePreds(window_, preds, np.array([b[i%48] for b in betas]), i, dType)

        final_preds[:, i] = copy.deepcopy(preds)

        i += 1

    if dType == 'humidity':
        final_mean_err = mean_absolute_error(hum_test, final_preds)
    elif dType == 'temperature':
        final_mean_err = mean_absolute_error(temp_test, final_preds)

    return final_preds, final_mean_err;


def predict_variance_inf_hrlvl(budget, train_means, train_vars, betas, dType):
    """Method to make predictions based on variance based active inference (Hour level model)."""        
    window_ = None
    i = 1

    final_preds = np.ones((50, 96))

    if budget > 0:
        window_ = np.argpartition(train_vars, -budget)[-budget:]
    else:
        window_ = np.array([], dtype=int)

    preds = copy.deepcopy(train_means)

    if dType == 'humidity':
        preds[window_] = copy.deepcopy(hum_test[window_, 0])
    elif dType == 'temperature':
        preds[window_] = copy.deepcopy(temp_test[window_, 0])

    final_preds[:, 0] = copy.deepcopy(preds)

    for t in global_times[1:]:
        #if budget > 0:
        #    window_ = np.argpartition(train_vars[t], -budget)[-budget:]
        #else:
        #    window_ = np.array([], dtype=int)

        preds = makePreds(window_, preds, betas, i, dType)

        final_preds[:, i] = copy.deepcopy(preds)

        i += 1

    if dType == 'humidity':
        final_mean_err = mean_absolute_error(hum_test, final_preds)
    elif dType == 'temperature':
        final_mean_err = mean_absolute_error(temp_test, final_preds)

    return final_preds, final_mean_err;

def trainModel(x, y, degree=1):
    """Self designed Explicit method to train the model using linear regression."""
    #poly = PolynomialFeatures(degree)
    #z = poly.fit_transform(x)
    #return np.dot(np.linalg.pinv(z), y)
    clf = Lasso(alpha=.5)
    clf.fit(x, y)
    return clf

def get_betas_hrlvl(data):
    """Uses Linear Regression to predict beta values for each sensor."""
    mat = makeMatrix(data) 
    #Train only for one day or 48 time stamps. :|
    x_train = mat[:, :-1].T
    y_train_set = mat[:, 1:]
    clfs = np.array([trainModel(x_train, y_train) for y_train in y_train_set])

    #poly = PolynomialFeatures(1)
    #res = np.array([np.array([clf.predict(x_train_elem.reshape(1, -1)) for x_train_elem in x_train]).T[0] for clf in clfs])
    #res_t = mat[:, 1:]
    #train_error = mean_absolute_error(res, res_t)

    return clfs

def trainModel_phase2(x, y, degree=1):
    """Self designed Explicit method to train the model using linear regression."""
    #poly = PolynomialFeatures(degree)
    #z = poly.fit_transform(x)
    #return np.dot(np.linalg.pinv(z), y)
    #clf = BernoulliRBM()
    #clf = LinearRegression()
    clf = Lasso(alpha=.5)
    clf.fit(x.reshape(-1, 1), y)
    return clf

def train_over_obs(arr):
    arr = np.array(arr)
    return trainModel_phase2(arr[:-1], arr[1:])

def get_betas_hrlvl_phase2(data):
    """Uses Linear Regression to predict beta values for each sensor."""
    ret = []
    betas = []
    for data_elem in data: #Iterating thorugh each sensor.
        tmp = []
        [[tmp.append(elem) for elem in v] for v in data_elem.values()]
        betas.append(train_over_obs(tmp))
    return np.array(betas)

def get_betas_daylvl_phase2(data):
    ret = []
    for d in data:
        od = collections.OrderedDict(sorted(d.items()))
        vals = np.array(od.values())
        thetas = np.array([train_over_obs(v) for v in vals])
        ret.append(thetas)
    return np.array(ret)


def predict_window_inf_hrlvl_phase2(budget, train_means, betas, dType):
    """Method to make predictions based on windowed active inference (Hour level model)."""        
    start_ = 0
    window_ = None
    i = 1

    final_preds = np.ones((50, 96))

    end_ = start_ + budget
    if end_ <= 50:
        window_ = np.arange(start_, end_)
    else:
        window_ = np.append(np.arange(start_, 50), np.arange(end_ % 50))

    start_ += budget
    start_ %= 50

    preds = copy.deepcopy(train_means)

    if dType == 'humidity':
        preds[window_] = copy.deepcopy(hum_test[window_, 0])
    elif dType == 'temperature':
        preds[window_] = copy.deepcopy(temp_test[window_, 0])

    final_preds[:, 0] = copy.deepcopy(preds)

    for t in global_times[1:]:
        end_ = start_ + budget
        if end_ <= 50:
            window_ = np.arange(start_, end_)
        else:
            window_ = np.append(np.arange(start_, 50), np.arange(end_ % 50))

        start_ += budget
        start_ %= 50

        preds = makePreds_phase2(window_, preds, betas, i, dType)

        final_preds[:, i] = copy.deepcopy(preds)

        i += 1

    if dType == 'humidity':
        final_mean_err = mean_absolute_error(hum_test, final_preds)
    elif dType == 'temperature':
        final_mean_err = mean_absolute_error(temp_test, final_preds)

    return final_preds, final_mean_err;

def predict_window_inf_daylvl_phase2(budget, train_means, betas, dType):
    start_ = 0
    window_ = None
    i = 1

    final_preds = np.ones((50, 96))

    end_ = start_ + budget
    if end_ <= 50:
        window_ = np.arange(start_, end_)
    else:
        window_ = np.append(np.arange(start_, 50), np.arange(end_ % 50))

    start_ += budget
    start_ %= 50

    preds = copy.deepcopy(train_means)

    if dType == 'humidity':
        preds[window_] = copy.deepcopy(hum_test[window_, 0])
    elif dType == 'temperature':
        preds[window_] = copy.deepcopy(temp_test[window_, 0])

    final_preds[:, 0] = copy.deepcopy(preds)

    for t in global_times[1:]:
        end_ = start_ + budget
        if end_ <= 50:
            window_ = np.arange(start_, end_)
        else:
            window_ = np.append(np.arange(start_, 50), np.arange(end_ % 50))

        start_ += budget
        start_ %= 50

        preds = makePreds_phase2(window_, preds, np.array([b[i%48] for b in betas]), i, dType)

        final_preds[:, i] = copy.deepcopy(preds)

        i += 1

    if dType == 'humidity':
        final_mean_err = mean_absolute_error(hum_test, final_preds)
    elif dType == 'temperature':
        final_mean_err = mean_absolute_error(temp_test, final_preds)

    return final_preds, final_mean_err;

def predict_variance_inf_hrlvl_phase2(budget, train_means, train_vars, betas, dType):
    """Method to make predictions based on variance based active inference (Hour level model)."""        
    window_ = None
    i = 1

    final_preds = np.ones((50, 96))

    if budget > 0:
        window_ = np.argpartition(train_vars[0.5], -budget)[-budget:]
    else:
        window_ = np.array([], dtype=int)

    preds = copy.deepcopy(train_means)

    if dType == 'humidity':
        preds[window_] = copy.deepcopy(hum_test[window_, 0])
    elif dType == 'temperature':
        preds[window_] = copy.deepcopy(temp_test[window_, 0])

    final_preds[:, 0] = copy.deepcopy(preds)

    for t in global_times[1:]:
        if budget > 0:
            window_ = np.argpartition(train_vars[t], -budget)[-budget:]
        else:
            window_ = np.array([], dtype=int)

        preds = makePreds_phase2(window_, preds, betas, i, dType)

        final_preds[:, i] = copy.deepcopy(preds)

        i += 1

    if dType == 'humidity':
        final_mean_err = mean_absolute_error(hum_test, final_preds)
    elif dType == 'temperature':
        final_mean_err = mean_absolute_error(temp_test, final_preds)

    return final_preds, final_mean_err;

def predict_variance_inf_daylvl_phase2(budget, train_means, train_vars, betas, dType):
    window_ = None
    i = 1

    final_preds = np.ones((50, 96))

    if budget > 0:
        window_ = np.argpartition(train_vars[0.5], -budget)[-budget:]
    else:
        window_ = np.array([], dtype=int)

    preds = copy.deepcopy(train_means)

    if dType == 'humidity':
        preds[window_] = copy.deepcopy(hum_test[window_, 0])
    elif dType == 'temperature':
        preds[window_] = copy.deepcopy(temp_test[window_, 0])

    final_preds[:, 0] = copy.deepcopy(preds)

    for t in global_times[1:]:
        if budget > 0:
            window_ = np.argpartition(train_vars[t], -budget)[-budget:]
        else:
            window_ = np.array([], dtype=int)

        preds = makePreds_phase2(window_, preds, np.array([b[i%48] for b in betas]), i, dType)

        final_preds[:, i] = copy.deepcopy(preds)

        i += 1

    if dType == 'humidity':
        final_mean_err = mean_absolute_error(hum_test, final_preds)
    elif dType == 'temperature':
        final_mean_err = mean_absolute_error(temp_test, final_preds)

    return final_preds, final_mean_err;


def makePreds_phase1(window_hum, window_temp, hum_train_means, temp_train_means, timeIdx, timeVal):
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


def predict_window_inf_phase1(budget, hum_train_means, temp_train_means):
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

        hum_pred, temp_pred = makePreds_phase1(window_hum, window_temp, hum_train_means, temp_train_means, i, t)

        hum_preds[:, i] = copy.deepcopy(hum_pred)
        temp_preds[:, i] = copy.deepcopy(temp_pred)
        
        i += 1

    hum_mean_err = mean_absolute_error(hum_test, hum_preds)
    temp_mean_err = mean_absolute_error(temp_test, temp_preds)

    return hum_preds, temp_preds, hum_mean_err, temp_mean_err


def predict_variance_inf_phase1(budget, hum_train_means, temp_train_means, hum_train_vars, temp_train_vars):
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

        hum_pred, temp_pred = makePreds_phase1(window_hum, window_temp, hum_train_means, temp_train_means, i, t)

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

    hum_train_means_hrlvl = compute_means_hrlvl(hum_train_data)
    hum_train_vars_hrlvl = compute_vars_hrlvl(hum_train_data)
    temp_train_means_hrlvl = compute_means_hrlvl(temp_train_data)
    temp_train_vars_hrlvl = compute_vars_hrlvl(temp_train_data)

    hum_train_means_daylvl = compute_means_daylvl(hum_train_data)
    hum_train_vars_daylvl = compute_vars_daylvl(hum_train_data)
    temp_train_means_daylvl = compute_means_daylvl(temp_train_data)
    temp_train_vars_daylvl = compute_vars_daylvl(temp_train_data)


    """Initializing the first row to be written on to the CSV files."""
    first_row = np.ones(97)
    tmp = np.sort(hum_train_data[0].keys())
    tmp = tmp[1:]
    tmp = np.append(tmp, 0.0)
    first_row[1:] = np.append(tmp, tmp)
    global_times = copy.deepcopy(first_row[1:])
    first_row = list(first_row)
    first_row[0] = 'sensors'

    hum_test = makeMatrix(hum_test_data)
    temp_test = makeMatrix(temp_test_data)

    budgets = [0, 5, 10, 20, 25]
    #budgets = [5, 10, 20, 25]

    hum_errors_win = []
    temp_errors_win = []
    hum_errors_var = []
    temp_errors_var = []

    """Computing predictions."""
    for budget in budgets:
        betas_hum_hrlvl = get_betas_hrlvl(hum_train_data)
        betas_temp_hrlvl = get_betas_hrlvl(temp_train_data)

        hum_preds_win_hrlvl, hum_mean_err_win_hrlvl_p3 = predict_window_inf_hrlvl(budget, hum_train_means_hrlvl, betas_hum_hrlvl, 'humidity')
        temp_preds_win_hrlvl, temp_mean_err_win_hrlvl_p3 = predict_window_inf_hrlvl(budget, temp_train_means_hrlvl, betas_temp_hrlvl, 'temperature')

        hum_preds_var_hrlvl, hum_mean_err_var_hrlvl_p3 = predict_variance_inf_hrlvl(budget, hum_train_means_hrlvl, hum_train_vars_hrlvl, betas_hum_hrlvl, 'humidity')
        temp_preds_var_hrlvl, temp_mean_err_var_hrlvl_p3 = predict_variance_inf_hrlvl(budget, temp_train_means_hrlvl, hum_train_vars_hrlvl, betas_temp_hrlvl, 'temperature')

        ########################

        hum_preds_win_phase1, temp_preds_win_phase1, hum_mean_err_win_phase1, temp_mean_err_win_phase1 = predict_window_inf_phase1(budget, hum_train_means_daylvl, temp_train_means_daylvl)
        hum_preds_var_phase1, temp_preds_var_phase1, hum_mean_err_var_phase1, temp_mean_err_var_phase1 = predict_variance_inf_phase1(budget, hum_train_means_daylvl, temp_train_means_daylvl, hum_train_vars_daylvl, temp_train_vars_daylvl)

        ########################

        betas_hum_hrlvl = get_betas_hrlvl_phase2(hum_train_data)
        betas_temp_hrlvl = get_betas_hrlvl_phase2(temp_train_data)

        hum_preds_win_hrlvl, hum_mean_err_win_hrlvl = predict_window_inf_hrlvl_phase2(budget, hum_train_means_hrlvl, betas_hum_hrlvl, 'humidity')
        temp_preds_win_hrlvl, temp_mean_err_win_hrlvl = predict_window_inf_hrlvl_phase2(budget, temp_train_means_hrlvl, betas_temp_hrlvl, 'temperature')

        hum_preds_var_hrlvl, hum_mean_err_var_hrlvl = predict_variance_inf_hrlvl_phase2(budget, hum_train_means_hrlvl, hum_train_vars_daylvl, betas_hum_hrlvl, 'humidity')
        temp_preds_var_hrlvl, temp_mean_err_var_hrlvl = predict_variance_inf_hrlvl_phase2(budget, temp_train_means_hrlvl, hum_train_vars_daylvl, betas_temp_hrlvl, 'temperature')
        

        betas_hum_daylvl = get_betas_daylvl_phase2(hum_train_data)
        betas_temp_daylvl = get_betas_daylvl_phase2(temp_train_data)

        hum_preds_win_daylvl, hum_mean_err_win_daylvl = predict_window_inf_daylvl_phase2(budget, hum_train_means_hrlvl, betas_hum_daylvl, 'humidity')
        temp_preds_win_daylvl, temp_mean_err_win_daylvl = predict_window_inf_daylvl_phase2(budget, temp_train_means_hrlvl, betas_temp_daylvl, 'temperature')

        hum_preds_var_daylvl, hum_mean_err_var_daylvl = predict_variance_inf_daylvl_phase2(budget, hum_train_means_hrlvl, hum_train_vars_daylvl, betas_hum_daylvl, 'humidity')
        temp_preds_var_daylvl, temp_mean_err_var_daylvl = predict_variance_inf_daylvl_phase2(budget, temp_train_means_hrlvl, hum_train_vars_daylvl, betas_temp_daylvl, 'temperature')



        print 'Budget =', budget
        print hum_mean_err_win_hrlvl, temp_mean_err_win_hrlvl
        print hum_mean_err_var_hrlvl, temp_mean_err_var_hrlvl, '\n'

        print '------------------------------------------------\n'

        write_data(hum_preds_win_hrlvl_p3, str('results/humidity/w' + str(budget) + '.csv'))
        write_data(hum_preds_var_hrlvl_p3, str('results/humidity/v' + str(budget) + '.csv'))

        write_data(temp_preds_win_hrlvl_p3, str('results/temperature/w' + str(budget) + '.csv'))
        write_data(temp_preds_var_hrlvl_p3, str('results/temperature/v' + str(budget) + '.csv'))

        """Plotting Histograms."""

        plt.xticks(range(8), ['Phase1-window', 'Phase1-variance', 'Phase2-h-window', 'Phase2-h-variance', 'Phase2-d-window', 'Phase2-d-variance', 'Phase3-window', 'Phase3-variance'])
        plt.bar(range(8), [hum_mean_err_win_phase1, hum_mean_err_var_phase1, hum_mean_err_win_hrlvl, hum_mean_err_var_hrlvl, hum_mean_err_win_daylvl, hum_mean_err_var_daylvl, hum_mean_err_win_hrlvl_p3, hum_mean_err_var_hrlvl_p3] , align='center', alpha=0.4)
        plt.xlabel('$Experiments$')
        plt.ylabel('$Mean Absolute Error$')
        plt.title("Humidity Error vs Aproaches with Budget=" + str(budget))
        plt.show()


        plt.xticks(range(8), ['Phase1-window', 'Phase1-variance', 'Phase2-h-window', 'Phase2-h-variance', 'Phase2-d-window', 'Phase2-d-variance', 'Phase3-window', 'Phase3-variance'])
        plt.bar(range(8), [temp_mean_err_win_phase1, temp_mean_err_var_phase1, temp_mean_err_win_hrlvl, temp_mean_err_var_hrlvl, temp_mean_err_win_daylvl, temp_mean_err_var_daylvl, temp_mean_err_win_hrlvl_p3, temp_mean_err_var_hrlvl_p3] , align='center', alpha=0.4)
        plt.xlabel('$Experiments$')
        plt.ylabel('$Mean Absolute Error$')
        plt.title("Temperature Error vs Aproaches with Budget=" + str(budget))
        plt.show()

        #plt.xticks(range(5), ['Phase1-window', 'Phase1-variance', 'Phase2-h-window', 'Phase2-h-variance', 'Phase2-d-window'])
        #plt.bar(range(5), [hum_mean_err_win_phase1, hum_mean_err_var_phase1, hum_mean_err_win_hrlvl, hum_mean_err_var_hrlvl, hum_mean_err_win_daylvl] , align='center', alpha=0.4)
        #plt.xlabel('$Experiments$')
        #plt.ylabel('$Mean Absolute Error$')
        #plt.title("Error vs Aproaches with Budget=" + str(budget) + "\n(Day Level variance inference experiment observations removed due to high error)")
        #plt.show()
    lol=0