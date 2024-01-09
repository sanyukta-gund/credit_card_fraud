from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('credit_card_fraud_final.pkl')  # Load your trained model

@app.route('/')
def index():
    return render_template('frontend.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract variables from the form
    merchant = request.form['merchant']
    category = request.form['category']
    amt = float(request.form['amt'])
    city = request.form['city']
    zip = float(request.form['zip'])
    lat = float(request.form['lat'])
    long = float(request.form['long'])
    city_pop = float(request.form['city_pop'])
    job = request.form['job']
    merch_lat = float(request.form['merch_lat'])
    merch_long = float(request.form['merch_long'])
    hour = float(request.form['hour'])
    day = float(request.form['day'])
    month = float(request.form['month'])
    hours_diff_bet_trans = float(request.form['hours_diff_bet_trans'])

    # Make predictions using your model
    prediction = model.predict([[merchant, category, amt, city, zip_code, lat, long, city_pop, job, merch_lat, merch_long, hour, day, month, hours_diff_bet_trans]])

    return render_template('frontend.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
