#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv('Company Stock and Investment.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.drop(['Gold Investments', 'Comp Stock', 'Other sharesInvestments'], axis=1,inplace=True)


# In[6]:


df.head()


# In[7]:


df.shape


# In[8]:


df.dtypes


# In[9]:


df.info()


# In[10]:


df.describe()


# In[11]:


df.isnull().sum()


# In[12]:


df['Date']=pd.to_datetime(df['Date'],errors='coerce')


# In[13]:


df.dtypes


# In[14]:


df.set_index('Date',inplace=True)


# In[15]:


df.head()


# In[16]:


df.tail()


# In[17]:


import numpy as np 
train_set, test_set= np.split(df, [int(.75 *len(df))])


# In[18]:


print(train_set, test_set)


# In[19]:


train = df[:1488]
test = df[-496:]
plt.plot(train)
plt.plot(test)


# In[20]:


#Check whether data is stationary or not
from statsmodels.tsa.stattools import adfuller


# In[21]:


result=adfuller(df['Oil Investments'])


# In[22]:


print("The values given as output by adfuller is : \n 'ADF Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used")


# In[24]:


#perform dickey fuller test  
def test_stationarity(timeseries):
   print("Results of dickey fuller test")
   adft = adfuller(timeseries['Oil Investments'],autolag='AIC')
   output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
   for key,values in adft[4].items():
       output['critical value (%s)'%key] =  values
   print(output)
   
test_stationarity(df)

Notice that the p-value is less than 0.05 so i can reject the Null hypothesis. Also, the test statistics is less than the critical values.therefore, the data is stationary.
# In[25]:


from pandas.plotting import autocorrelation_plot 
autocorrelation_plot(df['Oil Investments'])
plt.show()


# In[26]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# In[27]:


from statsmodels.tsa.stattools import acf,pacf
# we use d value here(data_log_shift)
acf = acf(df, nlags=40)
pacf= pacf(df, nlags=40,method='ols')#plot PACF
plt.subplot(121)
plt.plot(acf) 
plt.axhline(y=0,linestyle='-',color='blue')
plt.axhline(y=-1.96/np.sqrt(len(df)),linestyle='--',color='black')
plt.axhline(y=1.96/np.sqrt(len(df)),linestyle='--',color='black')
plt.title('Auto corellation function')
plt.tight_layout()#plot ACF
plt.subplot(122)
plt.plot(pacf) 
plt.axhline(y=0,linestyle='-',color='blue')
plt.axhline(y=-1.96/np.sqrt(len(df)),linestyle='--',color='black')
plt.axhline(y=1.96/np.sqrt(len(df)),linestyle='--',color='black')
plt.title('Partially auto corellation function')
plt.tight_layout()


# In[28]:


import statsmodels.api as sm


# In[29]:


model = sm.tsa.statespace.SARIMAX(train,order=(1,1,1),seasonal_order=(1,1,1,12))
model_fit = model.fit()


# In[30]:


model_fit.summary()


# In[33]:


pred = pd.DataFrame(model_fit.predict(n_periods = 496),index=test.index)
pred.columns = ['Predictions']
pred


# In[ ]:


df['forecast']=model_fit.predict(start=1773,end=1972,dynamic = True)
df[['Oil Investments','forecast']].plot(figsize=(12,8))


# In[ ]:


from pandas.tseries.offsets import DateOffset
future_dates= [df.index[-1]+ DateOffset(months=x) for x in range(0,24)]


# In[ ]:


future_datest_df= pd.DataFrame(index = future_dates [1:], columns=df.columns)


# In[ ]:


future_datest_df.tail()


# In[ ]:




