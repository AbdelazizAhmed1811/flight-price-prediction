# Flight Price Prediction Dashboard

An interactive data exploration and visualization dashboard built with Dash and Plotly, designed to analyze Indian domestic flight prices and answer key research questions about pricing behavior. The dashboard also integrates a trained Machine Learning model that predicts ticket prices based on user-selected flight details.

**Live Dashboard:** https://flight-price-prediction-1-v1q3.onrender.com/

---

## Project Structure

```
flight-price-prediction/
├── app.py                        
├── requirements.txt              
├── flight_price_model.pkl        
└── Clean_Dataset.csv             
```

**app.py**
The main application file. Contains the full Dash layout, all Plotly chart callbacks, the Price Predictor tab with interactive inputs, and the ML model prediction callback. Also defines the three custom scikit-learn transformer classes required to load the saved model pipeline.

**requirements.txt**
Lists all Python dependencies needed to run the application, including Dash, Plotly, pandas, numpy, scikit-learn, joblib, and gunicorn for deployment.

**flight_price_model.pkl**
The serialized trained machine learning pipeline. Contains the full preprocessing pipeline and the Random Forest Regressor model, saved using joblib from the training notebook.

**Clean_Dataset.csv**
The flight price dataset sourced from Kaggle (EaseMyTrip). Contains approximately 300,000 records of domestic Indian flights with features including airline, source city, destination city, departure time, arrival time, number of stops, flight class, duration, days left before departure, and ticket price.

---

## Research Questions

The dashboard is structured around five research questions. Each chart in the Research Questions tab directly addresses one of them.

**Q(A) Does price vary with Airlines?**
A box plot displays the full price distribution for each airline, showing the median, interquartile range, and outliers. This reveals a clear divide between premium carriers (Vistara, Air India) and budget carriers (IndiGo, SpiceJet, AirAsia, GO_FIRST).

**Q(B) How is the price affected when tickets are bought 1 or 2 days before departure?**
A line chart plots average ticket price against days remaining before the flight, split by class. A red shaded zone highlights the 1-2 day window to show the sharp last-minute price spike, while a green zone marks the optimal booking window of 15-49 days in advance.

**Q(C) Does ticket price change based on departure time and arrival time?**
Two grouped bar charts show average price across the six time slots (Early Morning, Morning, Afternoon, Evening, Night, Late Night) for both departure and arrival time separately, split by Economy and Business class.

**Q(D) How does the price change with change in Source and Destination?**
A heatmap displays the average ticket price for every source-destination route combination. Each cell shows the price in INR, with color intensity indicating how expensive the route is relative to others.

**Q(E) How does the ticket price vary between Economy and Business class?**
A violin plot with an embedded box plot shows the full price density and spread for both classes, making it easy to see that Business class prices form an entirely separate distribution from Economy.

---

## Machine Learning Model Integration

The dashboard includes a Price Predictor tab powered by a trained Random Forest Regressor.

**How it works in the dashboard:**
The user selects airline, seat class, source city, destination city, departure time, arrival time, number of stops, flight duration, and days left until the flight. On clicking Predict Price, the inputs are assembled into a dataframe and passed through the full pipeline to produce a price estimate displayed in INR along with a category label (Great Deal, Average Price, Above Average, Premium Fare).

A feature importance chart in the same tab shows that seat class and days left are the two most influential features for the model.

---

## Local Setup

**Requirements:** Python 3.9 or higher

**Step 1 - Clone the repository and switch to the dashboard branch:**
```bash
git clone https://github.com/AbdelazizAhmed1811/flight-price-prediction.git
cd flight-price-prediction
git checkout Flight_Dashboard_Dash
```

**Step 2 - Install dependencies:**
```bash
pip install -r requirements.txt
```

**Step 3 - Ensure the data and model files are present:**
Make sure both `Clean_Dataset.csv` and `flight_price_model.pkl` are in the same directory as `app.py`. These files are included in the repository.

**Step 4 - Run the app:**
```bash
python app.py
```

**Step 5 - Open in browser:**
Navigate to `http://127.0.0.1:8050` in your browser.

---

## Deployment

The dashboard is deployed on Render using the free tier.

**Platform:** Render (https://render.com)

**Settings used:**
- Language: Python
- Branch: `Flight_Dashboard_Dash`
- Build Command: `pip install -r requirements.txt`
- Start Command: `gunicorn app:server`
- Instance Type: Free

The `server = app.server` line in `app.py` exposes the Flask server instance that gunicorn uses as the entry point. The custom scikit-learn classes (`FeatureDropper`, `FlightOrdinalEncoder`, `Log1pTransformer`) are registered on the `__main__` module before loading the model to ensure joblib can deserialize the pipeline correctly in the gunicorn environment.

Any push to the `Flight_Dashboard_Dash` branch triggers an automatic redeploy on Render.

---

## Dataset

- Source: Kaggle - EaseMyTrip Flight Price Prediction
- Link: https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction
- Records: approximately 300,000
- Collection period: February to March 2022
- Coverage: 6 major Indian metro cities

---

## Team Contributions

| Name | Role |
|---|---|
| Abdelaziz Ahmed | EDA, Data Preprocessing, Machine Learning Model |
| Begol Osama | Dash and Plotly Dashboard, Model Integration, Deployment |
| Ahmed Reda | Power BI Dashboard, Documentation |
