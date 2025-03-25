## 🌦️ Real  Time Weather Prediction Model
This project predicts:
- **Rain Prediction** (Yes/No) using classification.
- **Future Temperature & Humidity** using regression.

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
- **Performance:**
    - Accuracy: **0.86 (86.3%)**
    - F1 Score: **0.58 (for class 1)**
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
City: karnataka,IN
current Temperature: 34°C
Feels Like: 32°C
Minimum Temp: 34°C
Maximum Temp: 34°C
Humidity: 25 %
Weather Prediction: overcast clouds
Rain Prediction: No

Future Temperature Predictions:
15:00: 31.2°C
16:00: 31.2°C
17:00: 31.2°C
18:00: 31.2°C
19:00: 31.2°C

Future Humidity Predictions:
15:00: 38.0%
16:00: 38.0%
17:00: 38.0%
18:00: 38.0%
19:00: 38.0%
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



## 📊 **Model Evaluation**

### 🎯 **Classification Metrics**
- **Accuracy:** 86.3%
- **F1 Score:** 0.58 (for rain prediction)
- **Precision/Recall:** Better at predicting "No Rain" than "Rain".




## 🤝 **Contributing**
Feel free to fork this project, open issues, and submit pull requests!



## 📧 **Contact**
- ✉️ Email: trangadiarudra26@gmail.com
- 🔗 GitHub: [Your GitHub](https://github.com/your-profile)

---
