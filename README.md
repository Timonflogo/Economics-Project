# Economics-Project
Student: Timon Florian Godt
School of Business and Social Sciences, Aarhus University

## Used Methods in Python Code Files
ARIMA(X)
SARIMA(X)
LSTM
CONV1D
DEEPNN

## Abstract
Europe is headed towards a major energy crisis which is very likely to affect Industries, Businesses, and certainly also households, as energy prices are rising and temperatures are falling.
This paper investigates the applicability and potential of advanced forecasting techniques such as
ARIMA, SARIMA and their regression versions of including exogenous variables, ARIMAX and
SARIMAX, as well as Recurrent Neural networks (RNNs) such as the infamous Long-Short-TermMemory (LSTM), on the electricity consumption of one single student household in Aarhus. The
Electricity consumption series was tested for correlations with meteorological features from the region and no significant relationships were able to be established for hourly frequency data. The
reasons for this are assumed to be the fluctuations in the timetables of the residents which adds
certain noise to the dependent series as well as the representation of a single household as opposed to
an aggregate of multiple households in the same area. The electricity consumption was subsequently
forecasted with hourly, daily, and monthly frequency, where it was found that ARIMAX, SARIMA,
and SARIMAX models performed superior to other more complex models such as the LSTM.

## Conclusions
In this paper, a comprehensive analysis of the performance of multiple advanced algorithms on a single residential student household’s electricity consumption in
Aarhus was conducted. As of the researchers’ knowledge, this is the first work given the previously established data
characteristics. It was found that in the given case of a
student household, no significant relationship with meteorological features such as the weather could be established in the case of hourly frequency of all series. This
insignificant relationship could result out of the fact that
only one single household is represented in the data, the
residents are students and have very fluctuate timetables
which increases the difficulty of establishing a seasonal
daily daily electricity consumption pattern, as well as
periods of vacation, illness, or other reasons of absence,
where no consumption occurs, all of which potentially
distort the relationship with the weather.
The best performing algorithms on the hourly and
daily forecasting challenges with a forecasting horizon of
168 and 30 observations, respectively, was a simple ARIMAX model which is a ARIMA Regression model, using
an exogenous array of dummy encoded variables, representing the seasonal pattern of hour and day_of_week.
The inclusion of such arrays significantly improved the
long term forecasts in the case of hourly predictions of 168
observations, while only including a seasonal component
in a SARIMA specification signaled a significant decay of
the seasonality after only a couple of observations. In the
monthly forecasting of the kWh series with a forecasting
horizon of 12, the ARIMAX model performed worse than
its more complex variants SARIMA and SARIMAX.
In all cases the LSTM model was the most difficult
and computationally expensive to train, with training
times that took far more time than the less complex models, and performed worse in terms of RMSE on the test
set predictions. However, this does not invalidate the
application of the LSTM on these problems. Previous
works have shown that correctly carried out hyperparameter tuning can significantly increase the performance of
LSTM models on similar data.
In general, forecasting a single residential electricity
consumption series is rather difficult to achieve with low
error, as most achieved forecasts on the test set were significantly over-underestimating the actual series.. However in some cases, the seasonal pattern was able to be
modeled by the ARIMAX, SARIMAX, and LSTM while
in other cases such as the monthly forecasting in section
4.3 the general decreasing trend from winter months to
summer months was detected.
This paper could be used by other researchers and individuals to gain insights into the performance related to
multiple advanced time series methods on similar data
and it is believed to spark further research interests
such as the inclusion of more exogenous features, perhaps based on geolocation, which account for the absence of the students at a given point in the future, as
it is believed to significantly improve the forecasting performance of the so far best performing ARIMAX and
SARIMAX models. Furthermore, the forecasts achieved
could be paired with auxiliary forecasts for the price of
electricity to calculate the the future costs that households will have to carry to keep their lights on, therefore,
enabling them to
