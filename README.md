## 🌦️ Real  Time Weather Prediction Model Deployed on django based web app
This project predicts:
- **Rain Prediction** (Yes/No) using classification.
- **Future Temperature & Humidity** using regression.
- This project is a Django-based web application that provides weather forecasts for a user-specified city. It integrates real-time weather data from the OpenWeatherMap API and uses machine learning (ML) models to predict whether it will rain, as well as future temperature and humidity values for the next 5 hours. The application also predicts daily minimum and maximum temperatures using historical data. 
---

## 📚 **Project Overview**

This project uses:
- **Classification Task:** Predicting `RainTomorrow` (Yes/No).
- **Regression Task:** Predicting future temperature and humidity for the next 5 hours.

---


### ** Fetching Weather Data from API**
- Uses OpenWeatherMap API to get real-time weather data.
- Retrieves:
    - Temperature, humidity, wind direction, wind speed, and pressure.

---

### 📊 **Feature Columns:**
- `MinTemp`, `MaxTemp`, `WindGustDir`, `WindGustSpeed`, `Humidity`, `Pressure`, `Temp`

### 🎯 **Target:**
- `RainTomorrow` – Encoded as 0 (No) and 1 (Yes).

---

### 🎛️ **3. Model Training**

#### 🌧️ **Rain Prediction Model (Classification)**
- Model: `RandomForestClassifier`
- Number of Estimators: `97`
- Test-Train Split: `80-20`
- Hyperparameters: `n_estimators=97`, `random_state=42`
- **Performance:**
    - Accuracy: **0.86 (86.3%)**
    - F1 Score: **0.5833333333333334 (for class 1)**
    - Classification Report:
    ```
                  precision    recall  f1-score   support
       0       0.86      0.98      0.92        57
       1       0.88      0.44      0.58        16
    ```


---

#### 🌡️ **Temperature & Humidity Prediction (Regression)**
- Model: `RandomForestRegressor`
- Number of Estimators: `100`
- **Training**:
    - Data: Historical weather data from `weather.csv`
    - Hyperparameters: `n_estimators=100`, `random_state=42`
- Continuous variables used for regression:
    - `Temp` → Predicts future temperature.
    - `Humidity` → Predicts future humidity.
- Regression data is prepared with a lag where:
    - X → Current value
    - Y → Next value (t+1 prediction)

---

## 📈 **Prediction Results**

### ✅ **Example Output:**
```
clouds
Rain Prediction: No

- **Current Weather**:
  - City: Mumbai, IN
  - Current Temperature: 28°C
  - Feels Like: 29°C
  - Minimum Temp: Predicted by ML (e.g., 26°C)
  - Maximum Temp: Predicted by ML (e.g., 30°C)
  - Humidity: 62%
  - Weather Prediction: Haze
  - Rain Prediction: YES

- **Future Temperature Predictions**:
  - 11:00: 27.7°C
  - 12:00: 27.6°C
  - 13:00: 27.5°C
  - 14:00: 27.4°C
  - 15:00: 27.3°C

- **Future Humidity Predictions**:
  - 11:00: 56.5%
  - 12:00: 56.4%
  - 13:00: 56.3%
  - 14:00: 56.2%
  - 15:00: 56.1%

```

---

## 🔥 **API Usage for Real-Time Data**

- API: [OpenWeatherMap](https://openweathermap.org/)
- Set your API key:
```python
API_KEY = 'your_api_key_here'
```
- Base URL:
```
BASE_URL = 'https://api.openweathermap.org/data/2.5/'
```



## Future Improvements

1. **Use a Time Series Model**:
   - Replace `RandomForestRegressor` with a time series model (e.g., Prophet, LSTM) for more accurate temperature and humidity predictions.

2. **Fetch Forecast Data**:
   - Use the OpenWeatherMap `/forecast` endpoint to get accurate daily min/max temperatures and future weather data, reducing reliance on ML for these values.

3. **Improve Rain Prediction**:
   - Add more features (e.g., recent precipitation, cloud cover trends) and use a time series model to improve the `RandomForestClassifier`’s performance.




## 📊 **Model Evaluation**

### 🎯 **Classification Metrics**
- **Accuracy:** 86.3%
- **F1 Score:** 0.58 (for rain prediction)
- **Precision/Recall:** Better at predicting "No Rain" than "Rain".

### Installation
1. **Clone the Repository**:
   ```
   git clone <repository-url>
   cd weather_forecast
   ```

2. **Create a Virtual Environment**:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```
   pip install requirements.txt
   ```

4. **Set Up the OpenWeatherMap API Key**:
   - Sign up at [OpenWeatherMap](https://openweathermap.org/) to get a free API key.
   - In `your_app/views.py`, replace the `API_KEY` with your key:
     
     ```python
     API_KEY = 'your-api-key-here'
     ```

5. **Prepare Historical Data**:
   - Place your `weather.csv` file in the project root directory.
   - Ensure it has the following columns: `Date`, `Temp`, `Humidity`, `Pressure`, `Clouds`, `MinTemp`, `MaxTemp`, `WindGustDir`, `WindGustSpeed`, `RainTomorrow`.
   - Update the `csv_path` in `views.py` to match the location of your `weather.csv`:
     
     ```python
     
     csv_path = os.path.join(r'path/to/your/weather.csv')
     ```

6. **Set Up Static Files**:
   - Ensure the following images are in `your_app/static/images/`: `rain.jpg`, `cloudy.jpg`, `sunny.jpg`, `snow.jpg`, `default1.jpg` to `default5.jpg`.
   - In `settings.py`, configure static files:
     
     ```python
     STATIC_URL = '/static/'
     STATICFILES_DIRS = [BASE_DIR / "static"]
     ```

7. **Update `urls.py`**:
   
   - In `weather_forecast/urls.py`, ensure static files are served:
     ```python
     from django.contrib import admin
     from django.urls import path, include
     from django.conf import settings
     from django.conf.urls.static import static

     urlpatterns = [
         path('admin/', admin.site.urls),
         path('', include('your_app.urls')),
     ] + static(settings.STATIC_URL, document_root=settings.STATICFILES_DIRS[0])
     ```
   - In `your_app/urls.py`:
     
     ```python
     from django.urls import path
     from . import views

     urlpatterns = [
         path('', views.predict_weather, name='predict_weather'),
     ]
     ```

9. **Run the Application**:
   ```
   python manage.py runserver
   ```
   - Access the app at `http://localhost:8000/`.

## Usage

1. Open the application in your browser (`http://localhost:8000/`).
2. Enter a city name (e.g., "Mumbai") in the search bar and submit the form.
3. View the current weather, rain prediction, future temperature/humidity predictions, and predicted min/max temperatures.
4. The background image will change based on the weather description (e.g., rain, cloud, sun, snow, or a random default image).


## 🤝 **Contributing**
Feel free to fork this project, open issues, and submit pull requests!



## 📧 **Contact**
- ✉️ Email: trangadiarudra26@gmail.com
- 🔗 GitHub: [Your GitHub](https://github.com/your-profile)

---
