from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Sample dataset
data = {
    'Rainfall': [800, 900, 750, 650, 700, 850, 950, 620, 670, 730],
    'Temperature': [30, 32, 31, 33, 34, 29, 28, 35, 36, 30],
    'Population_Growth': [1.2, 1.5, 1.3, 1.7, 2.0, 1.1, 1.0, 2.2, 2.5, 1.4],
    'Water_Availability': [500, 550, 480, 450, 460, 520, 580, 430, 440, 490]
}

df = pd.DataFrame(data)

# Train Model
X = df[['Rainfall', 'Temperature', 'Population_Growth']]
y = df['Water_Availability']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Store tanker orders
orders = []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction.html')
def prediction():
    return render_template('prediction.html')

@app.route('/sample_data.html')
def sample_data():
    return render_template('sample_data.html')

@app.route('/water_tanker.html')
def water_tanker():
    return render_template('water_tanker.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['input'].split(',')
    try:
        input_data = np.array([[float(data[0]), float(data[1]), float(data[2])]])
        predicted = model.predict(input_data)
        return jsonify({"response": f"Predicted Water Availability: {predicted[0]:.2f} liters"})
    except Exception as e:
        return jsonify({"response": "Invalid input. Use format: Rainfall, Temperature, Population Growth"}), 400

@app.route('/get_data')
def get_data():
    return jsonify(df.to_dict(orient='records'))

@app.route('/order', methods=['POST'])
def order():
    name = request.form['name']
    phone = request.form['phone']
    address = request.form['address']
    amount = request.form['amount']
    orders.append({"name": name, "phone": phone, "address": address, "amount": amount})
    return "Order placed successfully!", 200

if __name__ == '__main__':
    app.run(debug=True)
