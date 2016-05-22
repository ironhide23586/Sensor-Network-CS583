# Sensor-Network-CS583

* Contains code to model a sensor network comprised of temperature & humidity sensors.
* Data used to train probabilistic models obtained from Intel Lab Experiments
* Implemented in 3 phases.

##Phase 1 ##
* The objective in this phase was to develop a model which could use the observed values of temperature and humidities and use them in conjunction with the means and variances of the observations from each sensor, at different times obtained from the training data, to produce predictions of the outputs of those sensors for which the reading cannot be attained.
* The technique utilized in the code takes into account the hidden correlations between the temperature and humidity sensors to produce more accurate results.

##Phase 2 ##
* The objective in this phase was to develop a model which could use the observed values of temperature and
humidities and use them in conjunction with the means and variances of the observations from each sensor, at
different times obtained from the training data, to produce predictions of the outputs of those sensors for which the
reading cannot be attained.
In the code, multiple approaches were employed namely

> Windowed Active Inference with hour constant model parameters.

> Windowed Active Inference with day constant model parameters.
> Variance based Active Inference with hour constant model parameters.
> Variance based Active Inference with day constant model parameters.
In the hour constant parameter model, the model parameters are kept constant at the 0.5 hour level and in the day
constant model, the parameters are kept constant at the day level.
