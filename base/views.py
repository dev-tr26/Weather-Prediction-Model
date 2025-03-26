# from django.shortcuts import render
# from django.http import HttpResponse
# import requests, os
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from datetime import datetime,timedelta
# import pytz
# from sklearn.metrics import accuracy_score, classification_report,f1_score

# API_KEY = 'f6c53be4635a8e9567ce469338e955cd'

# BASE_URL='https://api.openweathermap.org/data/2.5/'


# # Create your views here.

# # Fetching weather data


# def get_curr_weather(city):
#   url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"   # constructing api req url
#   response = requests.get(url)                                       # send req to api
#   data = response.json()
#   return {
#       'city':data['name'],
#       'current_temp':round(data['main']['temp']),             # data['main'] to access particular part of data
#       'feels_like': round(data['main']['feels_like']),
#       'humidity': round(data['main']['humidity']),
#       'temp_min': round(data['main']['temp_min']),
#       'temp_max': round(data['main']['temp_max']),
#       'description': data['weather'][0]['description'],        # provide textual descr. of weather
#       'country': data['sys']['country'],
#       'wind_gust_dir': data['wind']['deg'],
#       'pressure': data['main']['pressure'],
#       'wind_gust_speed': data['wind']['speed'],
  
#       'clouds': data["clouds"]["all"],
      
#       'visibility': data["visibility"],
      
      
  
#   }



# # reading historical data

# def read_data(file_name):
#   df =pd.read_csv(file_name)
#   df =df.dropna()   #removing rows with missing val
#   df = df.drop_duplicates()
#   return df


# # preprocessing the data for training


# def preprocess_data(data):
#   le = LabelEncoder()
#   data['WindGustDir'] = le.fit_transform(data['WindGustDir'])
#   y = le.fit_transform(data['RainTomorrow'])

#   x = data[['MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed', 'Humidity', 'Pressure', 'Temp']]
#   print(f"Training features: {x.columns.tolist()}")

#   return x ,y ,le


# # Training the model



# def train_rain_prediction_model(x,y):
#   x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
#   model = RandomForestClassifier(n_estimators=97, random_state=42)
#   model.fit(x_train,y_train)

#   y_pred = model.predict(x_test)
#   accuracy = accuracy_score(y_test,y_pred)
#   f1_score_of_model = f1_score(y_test,y_pred)

#   print(f"Accuracy: {accuracy}\n")
#   print(f"F1 Score: {f1_score_of_model}\n")
#   print((f"Classification Report:{classification_report(y_test,y_pred)}"))

#   return model



# # Preparing Regression Data

# def prepare_regression_data(data,feature):     # feature specifies col name
#   x,y =[],[]                                   # x will store feature var, y target we want to predict
#   for i in range(len(data)-1):
#     x.append(data[feature].iloc[i])            # x has data val = 70,72,74,76,78

#     y.append(data[feature].iloc[i+1])          # y = 72,74,76,78  n-1 next data points for prediction

#   x = np.array(x).reshape(-1,1)              # convert x and y to numpy arrays x to 2d array with one col format required for regression models
#   y = np.array(y)
#   return x,y


# # Training Regression Model

# def train_regression_model(x,y):
#   model = RandomForestRegressor(n_estimators=100,random_state=42)
#   model.fit(x,y)
#   return model



# # predicting from trained regression model for current weather

# def weather_prediction(model,curr_data):
#   predictions = [curr_data]

#   for i in range(5):
#     next_val = model.predict(np.array([predictions[-1]]))

#     predictions.append(next_val[0])

#   return predictions[1:]




# def predict_weather(request):
#   if request.method == 'POST':
#     city = request.POST.get('city')
#     curr_weather = get_curr_weather(city)
    
#     csv_path = os.path.join(r'C:\Users\trang\env3\Scripts\weather_predict\weather.csv')
    
#     historical_data = read_data(csv_path)
#     x,y,le = preprocess_data(historical_data)
    
#     # x,y = split_data(historical_data)
    
#     rain_model = train_rain_prediction_model(x,y)

#     # mapping wind direction to compass points

#     wind_deg = curr_weather['wind_gust_dir'] % 360   # wind_direction_degree ensuresing within 0 to 360 degrees

#     #compass direction along with range of degrees associated with that direction
#     compass_points = [
#         ("N",0,11.25), ("NNE",11.25,33.75), ("NE",33.75,56.25),
#         ("ENE",56.25,78.75), ("E",78.75,101.25), ("ESE",101.25,123.75),
#         ("SE",123.75,146.25), ("SSE",146.25,168.75), ("S",168.75,191.25),
#         ("SSW",191.25,213.75), ("SW",213.75,236.25), ("WSW",236.25,258.75),
#         ("W",258.75,281.25), ("WNW",281.25,303.75), ("NW",303.75,326.25),
#         ("NNW",326.25,348.75)
#     ]

#     # we find compass directions that matches wind degrees


#     for point, start, end in compass_points:
#         if start <= wind_deg < end:
#           compass_direction = point
#           break

#     # we need to encode compass direction in a lang our model can understand

#     if compass_direction in le.classes_:
#       compass_direction_encoded = le.transform([compass_direction])[0]
#     else:
#       compass_direction_encoded = -1


#     curr_weather_data = {
#         'MinTemp': curr_weather['temp_min'],
#         'MaxTemp': curr_weather['temp_max'],
#         'WindGustDir': compass_direction_encoded,
#         'WindGustSpeed': curr_weather['wind_gust_speed'],           # to use this we need it in categorical form
#         'Humidity': curr_weather['humidity'],
#         'Pressure': curr_weather['pressure'],
#         'Temp': curr_weather['current_temp'],

#     }

#     # rainfall prediction

#     curr_df = pd.DataFrame([curr_weather_data])



#     rain_predict = rain_model.predict(curr_df)[0]

#     # preparing  and training regression model for temp and humidity

#     x_temp, y_temp = prepare_regression_data(historical_data,'Temp')
#     x_humidity, y_humidity = prepare_regression_data(historical_data,'Humidity')

#     temp_model = train_regression_model(x_temp,y_temp)
#     humidity_model = train_regression_model(x_humidity,y_humidity)


#     # predict future temp and humidity


#     future_temp = temp_model.predict([[curr_weather_data['Temp']]] * 5)
#     future_humidity = humidity_model.predict([[curr_weather_data['Humidity']]] * 5)


#     # preparing time series future prediction

#     timezone = pytz.timezone('Asia/Kolkata')
#     curr_time = datetime.now(timezone)
#     next_hr = curr_time + timedelta(hours=1)
#     next_hr = next_hr.replace(minute=0, second=0, microsecond =0 )

#     future_time = [(next_hr + timedelta(hours=i)).strftime("%H:00") for i in range(5)]

  
#     # storing each val in seperate var 
  
#     t1, t2, t3, t4, t5 = future_time 
#     temp1, temp2, temp3, temp4, temp5 = future_temp
#     h1, h2, h3, h4, h5 = future_humidity
    
#     context = {
#       "location": city,
#       'current_temp':curr_weather['temp'],             # data['main'] to access particular part of data
#       'feels_like': curr_weather['feels_like'],
#       'humidity': curr_weather['humidity'],
#       'temp_min': curr_weather['temp_min'],
#       'temp_max': curr_weather['temp_max'],
#       'clouds': curr_weather['clouds'],
#       'description' : curr_weather["description"],
#       'city': curr_weather["city"],
#       'country': curr_weather["country"],
      
#       'time': datetime.now(),
#       'date': datetime.now.strftime("%B %d, %Y"),
      
#       'Wind Speed': curr_weather['wind_gust_speed'],
#       'pressure': curr_weather['pressure'],
      
#       'visibility': curr_weather['visibility'],
      
#       't1':t1,
#       't2':t2,
#       't3':t3,
#       't4':t4,
#       't5':t5,

#       'temp1': f"{round(temp1, 1)}",
#       'temp2': f"{round(temp2, 1)}",
#       'temp3': f"{round(temp3, 1)}",
#       'temp4': f"{round(temp4, 1)}",
#       'temp5': f"{round(temp5, 1)}",
      
#       'h1': f"{round(h1, 1)}",
#       'h2': f"{round(h2, 1)}",
#       'h3': f"{round(h3, 1)}",
#       'h4': f"{round(h4, 1)}",
#       'h5': f"{round(h5, 1)}",
  
#     }
  
#     return render(request, "index.html", context)

#   return render(request, "index.html", context)
  
  
  
  
  

# # def home(request):
# #     return HttpResponse("hiie")
    
    
# your_app/views.py
from django.shortcuts import render
from django.http import HttpResponse
import requests, os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from datetime import datetime, timedelta
import pytz
from sklearn.metrics import accuracy_score, classification_report, f1_score

API_KEY = 'f6c53be4635a8e9567ce469338e955cd'
BASE_URL = 'https://api.openweathermap.org/data/2.5/'

# Fetching weather data
def get_curr_weather(city):
    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()
    return {
        'city': data['name'],
        'current_temp': round(data['main']['temp']),
        'feels_like': round(data['main']['feels_like']),
        'humidity': round(data['main']['humidity']),
        'temp_min': round(data['main']['temp_min']),
        'temp_max': round(data['main']['temp_max']),
        'description': data['weather'][0]['description'],
        'country': data['sys']['country'],
        'wind_gust_dir': data['wind']['deg'],
        'pressure': data['main']['pressure'],
        'wind_gust_speed': data['wind']['speed'],
        'clouds': data["clouds"]["all"],
        'visibility': data["visibility"],
    }

# Reading historical data
def read_data(file_name):
    df = pd.read_csv(file_name)
    df = df.dropna()
    df = df.drop_duplicates()
    return df

# Preprocessing the data for training
def preprocess_data(data):
    le = LabelEncoder()
    data['WindGustDir'] = le.fit_transform(data['WindGustDir'])
    y = le.fit_transform(data['RainTomorrow'])
    x = data[['MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed', 'Humidity', 'Pressure', 'Temp']]
    print(f"Training features: {x.columns.tolist()}")
    return x, y, le

# Training the rain prediction model
def train_rain_prediction_model(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=97, random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1_score_of_model = f1_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}\n")
    print(f"F1 Score: {f1_score_of_model}\n")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    return model

# Preparing regression data
def prepare_regression_data(data, feature):
    x, y = [], []
    for i in range(len(data) - 1):
        x.append(data[feature].iloc[i])
        y.append(data[feature].iloc[i + 1])
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)
    return x, y

# Training regression model
def train_regression_model(x, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x, y)
    return model

# Predicting from trained regression model for current weather
def weather_prediction(model, curr_data):
    predictions = [curr_data]
    for i in range(5):
        next_val = model.predict(np.array([predictions[-1]]).reshape(-1, 1))
        predictions.append(next_val[0])
    return predictions[1:]

# Main view to predict weather
def predict_weather(request):
    # Default context for GET request
    context = {
        "location": "Klang",
        'current_temp': 24.1,
        'feels_like': 23,
        'humidity': 52,
        'temp_min': 19,
        'temp_max': 29,
        'clouds': 92,
        'description': "Light Rain",
        'city': "Klang",
        'country': "MY",
        'time': datetime.now(),
        'date': datetime.now().strftime("%B %d, %Y"),
        'wind_speed': 213,
        'pressure': 1009,
        'visibility': 1000,
        't1': "12:00",
        't2': "13:00",
        't3': "14:00",
        't4': "15:00",
        't5': "16:00",
        'temp1': "22.1",
        'temp2': "23.3",
        'temp3': "19.0",
        'temp4': "24.5",
        'temp5': "24.5",
        'h1': "43.4",
        'h2': "46.8",
        'h3': "47.2",
        'h4': "47.7",
        'h5': "47.7",
    }

    if request.method == 'POST':  # Corrected from requests.method to request.method
        city = request.POST.get('city', 'Klang')  # Default to Klang if no city is provided
        try:
            curr_weather = get_curr_weather(city)
        except Exception as e:
            # Handle API errors (e.g., invalid city name)
            context['error'] = f"Error fetching weather data for {city}: {str(e)}"
            return render(request, "index.html", context)

        csv_path = os.path.join(r'C:\Users\trang\env3\Scripts\weather_predict\weather.csv')
        try:
            historical_data = read_data(csv_path)
        except Exception as e:
            # Handle CSV file errors
            context['error'] = f"Error reading historical data: {str(e)}"
            return render(request, "index.html", context)

        x, y, le = preprocess_data(historical_data)
        rain_model = train_rain_prediction_model(x, y)

        # Mapping wind direction to compass points
        wind_deg = curr_weather['wind_gust_dir'] % 360
        compass_points = [
            ("N", 0, 11.25), ("NNE", 11.25, 33.75), ("NE", 33.75, 56.25),
            ("ENE", 56.25, 78.75), ("E", 78.75, 101.25), ("ESE", 101.25, 123.75),
            ("SE", 123.75, 146.25), ("SSE", 146.25, 168.75), ("S", 168.75, 191.25),
            ("SSW", 191.25, 213.75), ("SW", 213.75, 236.25), ("WSW", 236.25, 258.75),
            ("W", 258.75, 281.25), ("WNW", 281.25, 303.75), ("NW", 303.75, 326.25),
            ("NNW", 326.25, 348.75)
        ]

        compass_direction = "N"  # Default
        for point, start, end in compass_points:
            if start <= wind_deg < end:
                compass_direction = point
                break

        if compass_direction in le.classes_:
            compass_direction_encoded = le.transform([compass_direction])[0]
        else:
            compass_direction_encoded = -1

        curr_weather_data = {
            'MinTemp': curr_weather['temp_min'],
            'MaxTemp': curr_weather['temp_max'],
            'WindGustDir': compass_direction_encoded,
            'WindGustSpeed': curr_weather['wind_gust_speed'],
            'Humidity': curr_weather['humidity'],
            'Pressure': curr_weather['pressure'],
            'Temp': curr_weather['current_temp'],
        }

        curr_df = pd.DataFrame([curr_weather_data])
        rain_predict = rain_model.predict(curr_df)[0]

        x_temp, y_temp = prepare_regression_data(historical_data, 'Temp')
        x_humidity, y_humidity = prepare_regression_data(historical_data, 'Humidity')

        temp_model = train_regression_model(x_temp, y_temp)
        humidity_model = train_regression_model(x_humidity, y_humidity)

        future_temp = weather_prediction(temp_model, curr_weather_data['Temp'])
        future_humidity = weather_prediction(humidity_model, curr_weather_data['Humidity'])

        timezone = pytz.timezone('Asia/Kolkata')
        curr_time = datetime.now(timezone)
        next_hr = curr_time + timedelta(hours=1)
        next_hr = next_hr.replace(minute=0, second=0, microsecond=0)

        future_time = [(next_hr + timedelta(hours=i)).strftime("%H:%M") for i in range(5)]

        t1, t2, t3, t4, t5 = future_time
        temp1, temp2, temp3, temp4, temp5 = future_temp
        h1, h2, h3, h4, h5 = future_humidity

        context = {
            "location": city,
            'current_temp': curr_weather['current_temp'],
            'feels_like': curr_weather['feels_like'],
            'humidity': curr_weather['humidity'],
            'temp_min': curr_weather['temp_min'],
            'temp_max': curr_weather['temp_max'],
            'clouds': curr_weather['clouds'],
            'description': curr_weather["description"],
            'city': curr_weather["city"],
            'country': curr_weather["country"],
            'time': datetime.now(),
            'date': datetime.now().strftime("%B %d, %Y"),  # Corrected datetime.no to datetime.now
            'wind_speed': curr_weather['wind_gust_speed'],
            'pressure': curr_weather['pressure'],
            'visibility': curr_weather['visibility'],
            't1': t1,
            't2': t2,
            't3': t3,
            't4': t4,
            't5': t5,
            'temp1': f"{round(temp1, 1)}",
            'temp2': f"{round(temp2, 1)}",
            'temp3': f"{round(temp3, 1)}",
            'temp4': f"{round(temp4, 1)}",
            'temp5': f"{round(temp5, 1)}",
            'h1': f"{round(h1, 1)}",
            'h2': f"{round(h2, 1)}",
            'h3': f"{round(h3, 1)}",
            'h4': f"{round(h4, 1)}",
            'h5': f"{round(h5, 1)}",
        }

    return render(request, "index.html", context)