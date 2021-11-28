---
layout: page
title: Supplementary Methods
permalink: /Methods/
---

__Supplementary methods__

*Python libraries and dependencies*

The following dependencies were used in the creation of this dashboard (see code snippet below)

<details>
<summary>CODE</summary>

{% highlight python %}
'pandas', 1.2.3
'numpy', 1.19.5
'matplotlib.pyplot', 3.4.1
'scipy', 1.6.2
'plotly', 5.3.1
'dash', 1.20.0
'dash_core_components', 1.16.0
'dash_html_components', 1.1.3
'requests', 2.25.1
'statsmodels', 0.11.0
'prophet', 1.0.1
'pmdarima' 1.81
'tensorflow' 2.4.1
{% endhighlight %}

</details>

*Data pre-processing*

Following anonymisation, referral data was uploaded as a pandas dataframe. Redundant columns, duplicates and erroneous entries were removed and all dates and times were transformed to python date-time data-types for further manipulation. Specialist working diagnoses are designated by the on-call neurosurgical registrar when receiving the referral and include a total of 138 different options. The diagnosis is based on the information received at the point of the referral and may be modified as further information is shared or after senior review. Specialist diagnoses were aggregated into 13 primary diagnostic categories: brain tumour, cauda equina syndrome, congenital, subdural haematoma, cranial trauma, degenerative spine, hydrocephalus, infection, spinal trauma, stroke, neurovascular and ‘not neurosurgical’ (Supplementary Appendix). 

<details>
<summary>CODE</summary>

{% highlight python %}
#Upload anonymised file - either saved as .csv or .pkl

df_all = pd.read_pickle(filename)

#Drop duplicates
df_all.drop_duplicates(inplace=True)

#Drop redundant columns
df_all.drop(columns = ['Referring Doctor Name','Bleep or Telephone No','MobileNo','Subsequent Doctor Grade Name','Subsequent Bleep Number','Subsequent Mobile No','Subsequent Dr Email Address','Subsequent Consultant Email Address'], inplace = True)

#Transform date-time entries to datetime datatype
df_all = transform_to_datetime(df_all, 'Referral Time')

#Convert specialist working diagnosis into primary diagnostic classification based on diagnosis table - see Appendix table
diagnosis_table = pd.read_csv('diagnoses_table.csv', low_memory=False)
df_all = add_classification_level(df_all, diagnosis_table,
                                  'Primary Classification')


## RELEVANT PROCESSING FUNCTIONS

def match_classification(diagnosis_table, classification_level,
                               diagnosis):
    diagnosis_level = diagnosis_table[
        diagnosis_table['Specialist working diagnosis'] ==
        diagnosis][classification_level]
    if (len(diagnosis_level.values) > 0):
        return diagnosis_level.values[0]
    return 'no_match'

def add_classification(input_df, diagnosis_table, classification_level):
    df_copy = copy.deepcopy(input_df)
    partial_func = partial(match_classification, diagnosis_table,
                           classification_level)
    df_copy[classification_level] = df_copy[
        'Specialist Working Diagnosis'].apply(partial_func)
    return df_copy
    
def transform_to_datetime(df, time_col):
    copy = df.copy()
    copy[time_col] = pd.to_datetime(copy[time_col], dayfirst=True)
    return copy
{% endhighlight %}

</details>

*Geographical information*

Using the name of the referring site, an application programming interface (API) request is made to *openstreetmap.org* to derive the latitude and longitude of referral site locations. This location data is then cached and parsed to a geographical plotting function.

<details>
<summary>CODE</summary>

{% highlight python %}
##API REQUEST TO GENERATE LATITUDE AND LONGITUDE CO-ORDINATES

def placemaker(df_all):
    
    #Parse and sort dataframe
    geocount = df_all
    geocount = geocount.groupby(by=['Primary Classification','Referring Hospital'])[['Age']].count().unstack(level=0)
    geocount.columns = geocount.columns.droplevel()
    geocount.fillna(value=0,inplace=True)
    geocount['total'] = geocount.sum(axis=1)
    geocount.reset_index(inplace = True)

    #Generate empty columns to fill location data in
    geocount['add'] = 0
    geocount['lon'] = 0
    geocount['lat'] = 0
    geocount = geocount.sort_values(by = 'total', ascending = False)
    geocount.reset_index(drop=True, inplace=True)

    #Generate list of unique hospitals from dataframe
    hosplist = geocount['Referring Hospital'].unique()
    hosplist = hosplist.tolist()

    #For each unique hospital, perform an API request
    for i,v in enumerate(hosplist):

        address = v
        url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote(address)+'?format=json'
        response = requests.get(url).json()
        geocount.loc[i,['add', 'lon', 'lat']] = [address,response[0]["lon"],response[0]["lat"]]
        
    #Create seperate dataframe to save location data to cache
    locmatch = pd.DataFrame()
    locmatch['Referring Hospital'] = geocount['add']
    locmatch['lon'] = geocount.lon
    locmatch['lat'] = geocount.lat
    locmatch.to_csv('locmatch2.csv')
        
    return geocount, hosplist, locmatch

##GENERATE GEOGRAPHICAL FIGURE

def geospatial(df, date1, date2,classification):
    
    #select data by time
    geocount = single_period(df, date1, date2)

    #filter df by primary classification and sort
    if classification != "all":
           geocount = geocount[geocount['Primary Classification'] == classification]

    geocount = geocount.groupby(by=['Primary Classification','Referring Hospital'])[['Age']].count().unstack(level=0)
    geocount.columns = geocount.columns.droplevel()
    geocount.fillna(value=0,inplace=True)
    geocount['total'] = geocount.sum(axis=1)
    geocount.reset_index(inplace = True)
    geocount = geocount.sort_values(by='total', ascending=False)
    geocount.reset_index(drop=True, inplace=True)
    geocount = geocount.merge(locmatch, on='Referring Hospital')

    #create figure, can be scaled by color or size
    fig5 = px.scatter_mapbox(geocount,
                             lat="lat",
                             lon="lon",
                             hover_name="Referring Hospital",
                             hover_data=["total"],
                             zoom=9,
                             height=300,
                             size=geocount.total,
                             size_max=40,
                             color="total",
                             center={
                                 'lat': 51.6,
                                 'lon': -0.26
                             },
                             opacity=0.7)

    #update layouts
    fig5.update_layout(mapbox_style='carto-positron')
    fig5['data'][0]['showlegend'] = False
    fig5['data'][0]['name'] = 'Referring Site'
    fig5.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig5.update_layout(autosize=True, width=800, height=800)

    return fig5


## RELEVANT PROCESSING FUNCTIONS

def single_period(df, date1, date2):
    return df[(df['Referral Time'] >= date1) & (df['Referral Time'] < date2)]


{% endhighlight %}

</details>

*Implementation of time-series forecasting models*

Three forecasting algorithms were trialled in this work: an automated pipeline which combined Seasonal and Trend decomposition using Loess (STL) with an automatic Autoregressive Integrated Moving Average (Auto-ARIMA) model, a Convolutional Neural Network - Long Short-Term Memory (CNN-LSTM) network and Prophet. In this section we describe how each model was implemented.

In preparation for time-series analysis, the referral volumes were sorted into weekly brackets, rather than taking daily volumes. This was to compensate for an observed ‘weekend’ effect seen in the daily referral data (see Figure 5A, Supplementary Table 1). 

__S Table 1 Median weekday and weekend volumes__ 
  Four highest referring categories are shown. p values shown are Bonferroni multiple comparison corrected following univariate Mann-Whitney U tests (NS = not significant).

| __Diagnostic Classification__ | __Median weekday volume__ | __Median weekend volume__ | __p__       |
|---------------------------|-----------------------|-----------------------|---------|
| All                       | 34.0                  | 17.5                  | <0.0001 |
| Brain tumour              | 6.8                   | 3.5                   | <0.0001 |
| Degenerative spine        | 4.6                   | 2.0                   | <0.0001 |
| Neurovascular             | 2.4                   | 2.0                   | 0.06    |
| Stroke                    | 2.2                   | 2.0                   | NS      |

We performed an exploratory analysis of the time-series using auto-correlation and partial auto-correlation plots in combination with augmented Dickey-Fuller testing to determine the degree of stationarity in the data and assist in defining initial parameters for seasonal decomposition and upper and lower limits for the auto-ARIMA grid search.
                                                                                      
ARIMA models are often considered a benchmark model in fields such as econometrics (Box / Jenkins). Here, two adjustments were made to enable automatic hyperparameter tuning and make the model robust to time-series of uncertain length, frame and degree of seasonality. First, a Seasonal and Trend decomposition using Loess (STL) was applied which separates the raw data into seasonal, trend and residual components. Each component is fed into an automated grid search to determine p, d and q parameters which describe the lag order, degree of differencing and order of moving average respectively. Optimal parameters are determined by minimisation of the Akaike Information Criterion (AIC) from the grid search and are used to fit the model. In this way if the seasonal and trend decomposition fails to enforce stationarity in the trend data (if for example there are multiple layers of seasonality), the auto-ARIMA step can separately model the trend, seasonality and residual before recomposing the data to forecast.

<details>
<summary>CODE</summary>

{% highlight python %}
### STL/Auto-ARIMA model
#Run EDA on weekly time-series first to manually check seasonality

#Set variables
res = []

#STL period corresponds to expected seasonality. 4 chosen to reflect monthly seasonal changes.
##Also can use 52 for yearly or 26 for 6-monthly seasonality
period = 4

#How long into future/out-of-sample to make forecast
future = 0
#95% Confidence interval
confidence = 0.05

#STL decomposition with default parameters and period - can be further tuned using grid search
res = STL(df, period = period, robust = False).fit()

#Seasonal auto-ARIMA, stepwise can be changed to True for more thorough grid search
smodel = pm.auto_arima(res.seasonal,
                   start_p=0, max_p=5,
                   start_q=0, max_q=5,
                   seasonal=False,
                   stepwise = False,
                   start_d=0, max_d=5,
                   trace=False, error_action='ignore');

#Trend auto-ARIMA
tmodel = pm.auto_arima(res.trend,
                   start_p=0, max_p=5,
                   start_q=0, max_q=5,
                   seasonal=False,
                   stepwise = False,
                   start_d=0, max_d=5,
                   trace=False, error_action='ignore');

#Residual auto-ARIMA
rmodel = pm.auto_arima(res.resid,
                   start_p=0, max_p=5,
                   start_q=0, max_q=5,
                   seasonal=False,
                   stepwise = False,
                   start_d=0, max_d=5,
                   trace=False, error_action='ignore');

#Modelling seasonality
modelsea = SARIMAX(res.seasonal, order = smodel.order, seasonal_order= smodel.seasonal_order).fit()

#If Auto-ARIMA fails then use simple differenced d=1 model for trend
try:
    modeltrend = ARIMA(res.trend, order = tmodel.order, freq=interval).fit()
except:
    modeltrend = ARIMA(res.trend, order = (0,1,0), freq=interval).fit()
    
#Modelling residual
modelres = ARIMA(res.resid, order = rmodel.order, freq=interval).fit()

#Forecasting and recomposition
forecast_season  = modelsea.forecast(future, alpha=confidence)
forecast_trend, std_err_trend, confidence_int_trend = modeltrend.forecast(future, alpha=confidence)
forecast_resid, std_err_resid, confidence_int_resid = modelres.forecast(future, alpha=confidence)
forecast_final = forecast_season + forecast_trend + forecast_resid
conf = confidence_int_trend + confidence_int_resid                                
{% endhighlight %}

</details>

Deep learning methods such as CNN and LSTM neural networks are able to discover and model hidden complexity within data and extract features of interest automatically. LSTM is a sub-type of recurrent neural network which is able to model lengthy temporal data. A CNN model can be used in a hybrid model with a LSTM network. Here the CNN is used to learn discriminative features by applying a non-linear transformation on the time-series data, which are then fed to the LSTM layers. 
We split the time-series into subsequences with 52 “steps” (i.e. one year) as the input and one output. This is then split into two sub-samples, each with two targets. This is passed into the convolutional layer which transforms the subsamples before downsampling, flattening and passing to a single LSTM layer with 64 neurons. Dropout proportion was set to 30%, in order to reduce overfitting. The number of filters in the convolutional layer, neurons and dropout proportion were selected following a hyperparameter grid search. For out-of-sample predictions longer than one week, the predicted value was used to iteratively increase the training set and the test set is therefore progressively used to fit the model.

<details>
<summary>CODE</summary>

{% highlight python %}
###CNN-LSTM implementation

#Relevant imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Flatten, TimeDistributed, Conv1D, MaxPooling1D

# define input sequence from dataframe
sequence = df['all'].to_list()

# Set number of steps, keep even
n_steps = 52

# split into an array of subsequences, X = input
X, y = sequence_split(sequence, n_steps)

features = 1
n_seq = 2

# divided subsequence into 2 subsamples
n_steps_2 = n_steps/2

# reshape input data for CNN layer
X = X.reshape((X.shape[0], n_seq, n_steps2, features))

# set up sequential stack model
model = Sequential()

#CNN layer with 64 output filters, kernel size corresponds to length of convolutional window
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, n_steps2, n_features)))

# Down samples by pool size
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))

#Flatten to single 1D vector
model.add(TimeDistributed(Flatten()))

#Single LSTM layer with 64 neurons
model.add(LSTM(64, activation='relu'))

#NN dense layer
model.add(Dense(1))

#ADAM optimisation using mse as a cost function
model.compile(optimizer='adam', loss='rmse')
model.fit(X, y, epochs=500, verbose=0)


## RELEVANT PROCESSING FUNCTIONS

def sequence_split(sequence, n_steps):
    
    #Prepare list variables
    X, y = list(), list()
    
    for i in range(len(sequence)):
        
        # find index at sequence end
        end_index = i + n_steps
        
        # stop code if has gone past total length of sequence
        if end_index > len(sequence)-1:
        break
        
        # divide sequence into subsamples
        seq_x, seq_y = sequence[i:end_index], sequence[end_index]
        X.append(seq_x)
        y.append(seq_y)
        
    return np.array(X), np.array(y)
{% endhighlight %}

</details>

Prophet is an open-source library provided by Facebook (https://facebook.github.io/prophet/). Prophet decomposes a time series into four components: growth, yearly and weekly seasonality and holidays, then fits an additive regression model [Taylor and Letham].  Growth is modelled as a piecewise linear or logistic growth trend, yearly seasonality is modelled using Fourier series, weekly seasonality is modelled using dummy variables, and holidays are inputted by the user. When modelling, Prophet automatically detects ‘changepoints’ in the trend. In applying this model, we performed a grid search hyperparameter tuning to identify the changepoint and seasonality prior scale and specified the lockdown period as a custom ‘holiday’.

<details>
<summary>CODE</summary>

{% highlight python %}
### Prophet implementation

#Specify dataframe and convert to prophet input

prophetdf = df.reset_index()
prophetdf.columns = ['ds', 'y']

#Specify weeks to predict
prediction = 1

#Specify lockdown period
lockdown = pd.DataFrame({
      'holiday': 'lockdown',
      'ds': pd.to_datetime(['2020-03-23']),
      'lower_window': 0,
      'upper_window': 84,
    })

#Set model parameters
model = Prophet(yearly_seasonality=True,
                weekly_seasonality=True,
                seasonality_mode='additive',
                interval_width=0.95,
                changepoint_prior_scale= 0.05,
                seasonality_prior_scale= 0.1,
                holidays = lockdown)

#Fit model
model.fit(prophetdf)
future = model.make_future_dataframe(periods=prediction,freq='W')

#Make predictions
forecast = model.predict(future)

{% endhighlight %}

</details>

*Usability, acceptability and feasibility*

This study employed a mixed-method design to assess dashboard usability, acceptability and feasibility. Participants were recruited from the local neurosurgical centre through mailing lists and were included if they had an adequate experience of using the electronic referral system (> 6 months). Participants were excluded if they were aware of the development of the dashboard. 

In each testing session, a demonstration of the dashboard’s capabilities were shown (~ 10-minutes). As an example which would simulate a typical service evaluation, participants were shown how to use features to audit a particular diagnostic category or time-period. Using a think-aloud protocol, participants were invited to explore the functions of the dashboard independently, after which they completed an electronic questionnaire that incorporated three validated instruments: the System Usability Scale (SUS), Acceptability of Intervention Measure (AIM) and Feasibility of Intervention Measure (FIM) adapted for use. The SUS asks participants to respond to a set of 10 statements using a 5 point Likert scale, with a composite score above 70 defined as “good” usability. 

In each of the AIM and FIM scales, participants were presented with 4 statements in reference to the ‘intervention’ (dashboard) and asked to rate these according to a 5-point Likert Scale. These statements have been previously assessed for substantive and discriminant content validity (Weiner). Two white-box questions were also incorporated into the questionnaire: “Which aspects or features of the dashboard did you find useful?” and “Do you have any suggestions for improving the dashboard?”. The questionnaire has been outlined in full in the Supplementary Appendix.
