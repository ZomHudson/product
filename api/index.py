from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from typing import Dict
import json
import os

app = Flask(__name__)
CORS(app)

# Use /tmp for serverless writable area
TMP_HISTORY_PATH = "/tmp/prediction_history.json"
TMP_CSV_PATH = "/tmp/ExFarmPrice.csv"

def ensure_tmp_csv_exists():
    """Copy CSV from project root to /tmp on cold start"""
    if not os.path.exists(TMP_CSV_PATH):
        try:
            # Try to find CSV in various locations
            possible_paths = [
                "ExFarmPrice.csv",
                "../ExFarmPrice.csv",
                os.path.join(os.path.dirname(__file__), "..", "ExFarmPrice.csv"),
                os.path.join(os.path.dirname(__file__), "ExFarmPrice.csv")
            ]
            
            csv_found = False
            for src_path in possible_paths:
                if os.path.exists(src_path):
                    with open(src_path, "rb") as fr:
                        data = fr.read()
                    with open(TMP_CSV_PATH, "wb") as fw:
                        fw.write(data)
                    csv_found = True
                    print(f"CSV copied from {src_path} to {TMP_CSV_PATH}")
                    break
            
            if not csv_found:
                # Create minimal template
                df = pd.DataFrame([{"Date_Range": "01.01.2025 - 07.01.2025", "Avg_Price": 6.50}])
                df.to_csv(TMP_CSV_PATH, index=False)
                print("Created default CSV template")
        except Exception as e:
            print(f"Error preparing tmp CSV: {e}")

class ChickenRestockPredictor:
    def __init__(self, api_url: str):
        self.api_url = api_url
        self.MIN_STOCK = 1000
        self.MAX_STOCK = 2000
        self.BASE_DEMAND = 1200
        self.RESTOCK_DAYS = [0, 3, 5]
        self.history_file = TMP_HISTORY_PATH
        ensure_tmp_csv_exists()
        self.load_history()

    def load_history(self):
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    self.history = json.load(f)
            else:
                self.history = []
        except Exception as e:
            print(f"Error loading history: {e}")
            self.history = []

    def save_history(self):
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print(f"Error saving history: {e}")

    def add_historical_record(self, prediction: Dict, actual: int = None):
        record = {
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction,
            'actual': actual
        }
        self.history.append(record)
        self.save_history()

    def get_historical_accuracy(self, days: int = 30):
        if not self.history:
            return None

        cutoff_date = datetime.now() - timedelta(days=days)
        recent_history = [
            h for h in self.history
            if datetime.fromisoformat(h['timestamp']) > cutoff_date
            and h.get('actual') is not None
        ]

        if not recent_history:
            return None

        accuracies = []
        for record in recent_history:
            predicted = record['prediction']['predicted_quantity']
            actual = record['actual']
            if actual > 0:
                accuracy = 100 - abs((predicted - actual) / actual * 100)
                accuracies.append(max(0, accuracy))

        if accuracies:
            return {
                'avg_accuracy': float(np.mean(accuracies)),
                'min_accuracy': float(np.min(accuracies)),
                'max_accuracy': float(np.max(accuracies)),
                'total_predictions': len(recent_history)
            }
        return None

    def fetch_current_stock(self) -> Dict:
        try:
            response = requests.get(self.api_url, timeout=10)
            data = response.json()

            if data.get('success', False):
                factory_stock = 0
                kiosk_stock = 0

                for item in data.get('factory_data', []):
                    if item['item_name'] == 'Marinated Chicken':
                        factory_stock = int(item['stock_count'])
                        break

                for kiosk in data.get('kiosk_data', []):
                    for item in kiosk.get('items', []):
                        if item['item_name'] == 'Marinated Chicken':
                            kiosk_stock += int(item['stock_count'])

                return {
                    'factory_stock': factory_stock,
                    'kiosk_stock': kiosk_stock
                }
            else:
                raise Exception("API returned unsuccessful response")

        except Exception as e:
            print(f"Error fetching stock data: {e}")
            return {'factory_stock': 0, 'kiosk_stock': 0}

    def parse_date_range(self, date_str):
        try:
            parts = date_str.split(' - ')
            if len(parts) == 2:
                end_date_str = parts[1].strip()
                return datetime.strptime(end_date_str, '%d.%m.%Y')
            return None
        except:
            return None

    def get_current_price(self) -> float:
        try:
            df = pd.read_csv(TMP_CSV_PATH)
            df['end_date'] = df['Date_Range'].apply(self.parse_date_range)
            df = df.dropna(subset=['end_date'])
            latest_row = df.sort_values('end_date', ascending=False).iloc[0]
            return float(latest_row['Avg_Price'])
        except Exception as e:
            print(f"Error reading price data: {e}")
            return 6.5

    def _get_seasonal_price_factor(self, target_date: datetime) -> float:
        month = target_date.month
        day = target_date.day

        if (month == 1 and day > 15) or (month == 2 and day < 15):
            days_to_cny = abs((datetime(2025, 1, 29) - target_date).days)
            if days_to_cny < 7:
                return 0.15
            elif days_to_cny < 14:
                return 0.10
            else:
                return 0.05
        elif month in [3, 4]:
            if month == 3 and day > 20:
                return 0.15
            elif month == 4 and day < 5:
                return 0.12
            else:
                return 0.08
        elif month == 6:
            return 0.06
        elif month in [11, 12]:
            if month == 12 and day > 15:
                return 0.10
            else:
                return 0.06
        else:
            return 0.0

    def get_price_forecast(self, target_date: datetime) -> Dict:
        try:
            df = pd.read_csv(TMP_CSV_PATH)
            df['end_date'] = df['Date_Range'].apply(self.parse_date_range)
            df = df.dropna(subset=['end_date'])
            df = df.sort_values('end_date')

            if len(df) == 0:
                return {
                    'forecasted_price': 6.5,
                    'confidence': 'Low',
                    'method': 'fallback',
                    'factors': {}
                }

            current_price = float(df.iloc[-1]['Avg_Price'])
            days_ahead = (target_date - datetime.now()).days

            if days_ahead <= 3:
                return {
                    'forecasted_price': current_price,
                    'confidence': 'High',
                    'method': 'current',
                    'factors': {'base_price': current_price, 'days_ahead': days_ahead}
                }

            weeks_back = min(8, len(df))
            recent_df = df.tail(weeks_back)

            trend_adjustment = 0.0
            if len(recent_df) >= 3:
                price_changes = recent_df['Avg_Price'].pct_change().dropna()
                avg_weekly_change = price_changes.mean()
                weeks_ahead = days_ahead / 7
                trend_adjustment = avg_weekly_change * weeks_ahead
                trend_adjustment = max(-0.15, min(0.15, trend_adjustment))

            seasonal_factor = self._get_seasonal_price_factor(target_date)

            if len(recent_df) >= 3:
                price_volatility = recent_df['Avg_Price'].std() / recent_df['Avg_Price'].mean()
                if price_volatility < 0.05:
                    confidence = 'High'
                elif price_volatility < 0.10:
                    confidence = 'Medium'
                else:
                    confidence = 'Low'
            else:
                confidence = 'Low'

            total_adjustment = trend_adjustment + seasonal_factor
            forecasted_price = current_price * (1 + total_adjustment)
            forecasted_price = max(5.0, min(9.0, forecasted_price))
            forecasted_price = round(forecasted_price, 2)

            return {
                'forecasted_price': forecasted_price,
                'confidence': confidence,
                'method': 'trend_seasonal',
                'factors': {
                    'base_price': round(current_price, 2),
                    'trend_adjustment': round(trend_adjustment * 100, 2),
                    'seasonal_adjustment': round(seasonal_factor * 100, 2),
                    'total_adjustment': round(total_adjustment * 100, 2),
                    'days_ahead': days_ahead,
                    'weeks_of_data': weeks_back
                }
            }

        except Exception as e:
            print(f"Error forecasting price: {e}")
            return {
                'forecasted_price': 6.5,
                'confidence': 'Low',
                'method': 'fallback_error',
                'factors': {'error': str(e)}
            }

    def get_price_adjustment_factor(self, price: float) -> float:
        NORMAL_PRICE = 6.5
        HIGH_PRICE_THRESHOLD = 6.8
        LOW_PRICE_THRESHOLD = 6.2

        if price >= HIGH_PRICE_THRESHOLD:
            return 0.3
        elif price >= NORMAL_PRICE:
            return 0.15
        elif price <= LOW_PRICE_THRESHOLD:
            return -0.2
        else:
            return 0.0

    def get_calendar_events(self, target_date: datetime) -> Dict:
        calendar_events = {
            '2024-12-25': {'name': 'Christmas', 'factor': 0.30, 'pre_days': 3, 'post_days': 1},
            '2025-01-01': {'name': 'New Year', 'factor': 0.25, 'pre_days': 2, 'post_days': 1},
            '2025-01-29': {'name': 'Chinese New Year', 'factor': 0.40, 'pre_days': 5, 'post_days': 2},
            '2025-01-30': {'name': 'Chinese New Year (Day 2)', 'factor': 0.35, 'pre_days': 4, 'post_days': 2},
            '2025-03-31': {'name': 'Hari Raya Aidilfitri', 'factor': 0.50, 'pre_days': 5, 'post_days': 2},
            '2025-04-01': {'name': 'Hari Raya Aidilfitri (Day 2)', 'factor': 0.45, 'pre_days': 4, 'post_days': 2},
            '2025-05-01': {'name': 'Labour Day', 'factor': 0.20, 'pre_days': 1, 'post_days': 0},
            '2025-05-12': {'name': 'Wesak Day', 'factor': 0.15, 'pre_days': 1, 'post_days': 0},
            '2025-06-07': {'name': 'Hari Raya Aidiladha', 'factor': 0.30, 'pre_days': 3, 'post_days': 1},
            '2025-08-31': {'name': 'Merdeka Day', 'factor': 0.20, 'pre_days': 2, 'post_days': 0},
            '2025-09-16': {'name': 'Malaysia Day', 'factor': 0.20, 'pre_days': 2, 'post_days': 0},
            '2025-10-20': {'name': 'Deepavali', 'factor': 0.25, 'pre_days': 3, 'post_days': 1},
            '2025-12-25': {'name': 'Christmas', 'factor': 0.30, 'pre_days': 3, 'post_days': 1},
            '2026-01-01': {'name': 'New Year', 'factor': 0.25, 'pre_days': 2, 'post_days': 1},
        }
        
        ramadan_periods = [
            {'start': datetime(2025, 3, 1), 'end': datetime(2025, 3, 30), 'factor': 0.15}
        ]
        
        school_holidays = [
            {'start': datetime(2024, 11, 23), 'end': datetime(2025, 1, 5), 'name': 'Year End', 'factor': 0.15},
            {'start': datetime(2025, 3, 22), 'end': datetime(2025, 3, 30), 'name': 'Mid Year', 'factor': 0.12},
            {'start': datetime(2025, 5, 24), 'end': datetime(2025, 6, 8), 'name': 'Mid Year', 'factor': 0.12},
            {'start': datetime(2025, 8, 16), 'end': datetime(2025, 8, 24), 'name': 'Short Break', 'factor': 0.10},
            {'start': datetime(2025, 11, 22), 'end': datetime(2026, 1, 4), 'name': 'Year End', 'factor': 0.15},
        ]

        target_date_str = target_date.strftime('%Y-%m-%d')

        if target_date_str in calendar_events:
            event = calendar_events[target_date_str]
            return {'has_event': True, 'event_name': event['name'], 'factor': event['factor'], 'type': 'festival'}

        for date_str, event in calendar_events.items():
            event_date = datetime.strptime(date_str, '%Y-%m-%d')
            days_before = (event_date - target_date).days

            if 0 < days_before <= event['pre_days']:
                proximity_factor = event['factor'] * (1 - days_before / event['pre_days']) * 0.7
                return {'has_event': True, 'event_name': f"{days_before} days before {event['name']}", 'factor': proximity_factor, 'type': 'pre-festival'}

            days_after = (target_date - event_date).days
            if 0 < days_after <= event['post_days']:
                return {'has_event': True, 'event_name': f"{days_after} days after {event['name']}", 'factor': -0.25, 'type': 'post-festival'}

        for period in ramadan_periods:
            if period['start'] <= target_date <= period['end']:
                days_to_end = (period['end'] - target_date).days
                if days_to_end <= 14:
                    ramadan_factor = 0.15 + (14 - days_to_end) / 14 * 0.20
                    return {'has_event': True, 'event_name': f'Ramadan (approaching Raya)', 'factor': ramadan_factor, 'type': 'ramadan'}
                else:
                    return {'has_event': True, 'event_name': 'Ramadan', 'factor': 0.10, 'type': 'ramadan'}

        for holiday in school_holidays:
            if holiday['start'] <= target_date <= holiday['end']:
                return {'has_event': True, 'event_name': f"School Holiday ({holiday['name']})", 'factor': holiday['factor'], 'type': 'school-holiday'}

        if target_date.weekday() == 4:
            return {'has_event': True, 'event_name': 'Friday (weekend preparation)', 'factor': 0.12, 'type': 'friday'}

        return {'has_event': False, 'event_name': 'Normal day', 'factor': 0.0, 'type': 'normal'}

    def calculate_inventory_factor(self, factory_stock: int, kiosk_stock: int) -> float:
        total_stock = factory_stock + kiosk_stock

        if total_stock < 100:
            return 0.5
        elif total_stock < 300:
            return 0.3
        elif total_stock < 500:
            return 0.1
        elif total_stock > 1500:
            return -0.3
        elif total_stock > 1000:
            return -0.15
        else:
            return 0.0

    def calculate_day_of_week_factor(self, target_date: datetime) -> float:
        weekday = target_date.weekday()

        if weekday == 5:
            return 0.15
        elif weekday == 3:
            return 0.05
        elif weekday == 0:
            return 0.0
        else:
            return 0.0

    def predict_restock_quantity(self, target_date: datetime = None) -> Dict:
        if target_date is None:
            today = datetime.now()
            for i in range(7):
                check_date = today + timedelta(days=i)
                if check_date.weekday() in self.RESTOCK_DAYS:
                    target_date = check_date
                    break

        stock_data = self.fetch_current_stock()

        days_ahead = (target_date - datetime.now()).days
        if days_ahead > 3:
            price_info = self.get_price_forecast(target_date)
            current_price = price_info['forecasted_price']
            price_source = 'forecasted'
        else:
            current_price = self.get_current_price()
            price_info = {
                'forecasted_price': current_price,
                'confidence': 'High',
                'method': 'current',
                'factors': {'base_price': current_price}
            }
            price_source = 'current'

        inventory_factor = self.calculate_inventory_factor(
            stock_data['factory_stock'],
            stock_data['kiosk_stock']
        )

        price_factor = self.get_price_adjustment_factor(current_price)
        calendar_info = self.get_calendar_events(target_date)
        calendar_factor = calendar_info['factor']
        day_factor = self.calculate_day_of_week_factor(target_date)

        total_adjustment = inventory_factor + price_factor + calendar_factor + day_factor
        predicted_quantity = self.BASE_DEMAND * (1 + total_adjustment)
        predicted_quantity = max(self.MIN_STOCK, min(self.MAX_STOCK, predicted_quantity))
        predicted_quantity = round(predicted_quantity / 50) * 50

        result = {
            'target_date': target_date.strftime('%Y-%m-%d (%A)'),
            'predicted_quantity': int(predicted_quantity),
            'current_stock': {
                'factory': stock_data['factory_stock'],
                'kiosk': stock_data['kiosk_stock'],
                'total': stock_data['factory_stock'] + stock_data['kiosk_stock']
            },
            'factors': {
                'inventory_adjustment': f"{inventory_factor:+.2f} ({inventory_factor*100:+.1f}%)",
                'price_adjustment': f"{price_factor:+.2f} ({price_factor*100:+.1f}%)",
                'calendar_adjustment': f"{calendar_factor:+.2f} ({calendar_factor*100:+.1f}%)",
                'day_of_week_adjustment': f"{day_factor:+.2f} ({day_factor*100:+.1f}%)",
                'total_adjustment': f"{total_adjustment:+.2f} ({total_adjustment*100:+.1f}%)"
            },
            'calendar_event': calendar_info,
            'price_info': {
                'price': current_price,
                'source': price_source,
                'confidence': price_info['confidence'],
                'method': price_info['method'],
                'forecast_factors': price_info.get('factors', {})
            },
            'base_demand': self.BASE_DEMAND,
            'confidence': self._calculate_confidence(total_adjustment, price_info['confidence'])
        }

        return result

    def _calculate_confidence(self, total_adjustment: float, price_confidence: str) -> str:
        if abs(total_adjustment) > 0.5:
            quantity_confidence = "Medium"
        elif abs(total_adjustment) > 0.3:
            quantity_confidence = "Medium-High"
        else:
            quantity_confidence = "High"

        confidence_scores = {
            'High': 3,
            'Medium-High': 2,
            'Medium': 1,
            'Low': 0
        }

        qty_score = confidence_scores.get(quantity_confidence, 1)
        price_score = confidence_scores.get(price_confidence, 1)
        avg_score = (qty_score + price_score) / 2

        if avg_score >= 2.5:
            return "High"
        elif avg_score >= 1.5:
            return "Medium-High"
        elif avg_score >= 0.5:
            return "Medium"
        else:
            return "Low"

    def predict_next_week(self) -> list:
        predictions = []
        today = datetime.now()

        for i in range(14):
            check_date = today + timedelta(days=i)
            if check_date.weekday() in self.RESTOCK_DAYS:
                result = self.predict_restock_quantity(check_date)
                predictions.append(result)

        return predictions

    def get_price_history(self, days: int = 90) -> Dict:
        try:
            df = pd.read_csv(TMP_CSV_PATH)
            df['end_date'] = df['Date_Range'].apply(self.parse_date_range)
            df = df.dropna(subset=['end_date'])
            df = df.sort_values('end_date')

            cutoff_date = datetime.now() - timedelta(days=days)
            recent_df = df[df['end_date'] > cutoff_date]

            history = []
            for _, row in recent_df.iterrows():
                history.append({
                    'date': row['end_date'].strftime('%Y-%m-%d'),
                    'price': float(row['Avg_Price'])
                })

            today = datetime.now()
            for i in range(1, 15):
                forecast_date = today + timedelta(days=i)
                price_info = self.get_price_forecast(forecast_date)
                history.append({
                    'date': forecast_date.strftime('%Y-%m-%d'),
                    'price': price_info['forecasted_price'],
                    'is_forecast': True,
                    'confidence': price_info['confidence']
                })

            return {
                'success': True,
                'data': history,
                'current_price': float(recent_df.iloc[-1]['Avg_Price']),
                'avg_price': float(recent_df['Avg_Price'].mean()),
                'min_price': float(recent_df['Avg_Price'].min()),
                'max_price': float(recent_df['Avg_Price'].max())
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

# Initialize predictor
predictor = ChickenRestockPredictor(
    api_url="https://kimiez-storage.vercel.app/api/analytics/96e27e560a23a5a21978005c3d69add802bfa5b9be3cb6c1f7735e51db80bfe2/overview"
)

# API Routes
@app.route('/api/predict', methods=['GET'])
def get_prediction():
    try:
        result = predictor.predict_restock_quantity()
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/predict/week', methods=['GET'])
def get_weekly_predictions():
    try:
        predictions = predictor.predict_next_week()
        return jsonify({'success': True, 'data': predictions})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/price/current', methods=['GET'])
def get_current_price():
    try:
        price = predictor.get_current_price()
        return jsonify({'success': True, 'data': {'price': price, 'source': 'csv'}})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/price/forecast', methods=['GET'])
def get_price_forecast_route():
    try:
        date_str = request.args.get('date')
        if date_str:
            target_date = datetime.strptime(date_str, '%Y-%m-%d')
        else:
            target_date = datetime.now() + timedelta(days=7)

        price_info = predictor.get_price_forecast(target_date)
        return jsonify({'success': True, 'data': price_info})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/price/history', methods=['GET'])
def get_price_history():
    try:
        days = int(request.args.get('days', 90))
        result = predictor.get_price_history(days)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/price/update', methods=['POST'])
def update_price():
    try:
        data = request.json
        new_price = float(data.get('price'))
        date_range = data.get('date_range')

        if not date_range or new_price is None:
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400

        df = pd.read_csv(TMP_CSV_PATH)
        new_row = pd.DataFrame([{'Date_Range': date_range, 'Avg_Price': new_price}])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(TMP_CSV_PATH, index=False)

        return jsonify({
            'success': True,
            'message': 'Price updated successfully',
            'new_price': new_price,
            'date_range': date_range
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    try:
        days = int(request.args.get('days', 30))
        cutoff_date = datetime.now() - timedelta(days=days)

        recent_history = [
            h for h in predictor.history
            if datetime.fromisoformat(h['timestamp']) > cutoff_date
        ]

        return jsonify({'success': True, 'data': recent_history})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/accuracy', methods=['GET'])
def get_accuracy():
    try:
        days = int(request.args.get('days', 30))
        accuracy = predictor.get_historical_accuracy(days)

        if accuracy:
            return jsonify({'success': True, 'data': accuracy})
        else:
            return jsonify({'success': True, 'data': None, 'message': 'Insufficient historical data'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/record', methods=['POST'])
def record_actual():
    try:
        data = request.json
        date = data.get('date')
        actual_quantity = data.get('actual_quantity')

        if not date or actual_quantity is None:
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400

        target_date = datetime.strptime(date, '%Y-%m-%d')
        prediction = predictor.predict_restock_quantity(target_date)
        predictor.add_historical_record(prediction, actual_quantity)

        return jsonify({'success': True, 'message': 'Actual quantity recorded'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    try:
        prediction = predictor.predict_restock_quantity()
        stock = prediction['current_stock']['total']
        alerts = []

        if stock < 300:
            alerts.append({
                'type': 'critical',
                'message': 'Critical stock level detected',
                'detail': f"Current stock ({stock}) is below minimum threshold of 300 units",
                'timestamp': datetime.now().isoformat()
            })

        if stock < 500:
            alerts.append({
                'type': 'warning',
                'message': 'Low stock warning',
                'detail': f"Stock level at {stock} units. Consider restocking soon.",
                'timestamp': datetime.now().isoformat()
            })

        if prediction['predicted_quantity'] >= 1800:
            alerts.append({
                'type': 'info',
                'message': 'High demand period approaching',
                'detail': f"Predicted restock: {prediction['predicted_quantity']} units due to {prediction['calendar_event']['event_name']}",
                'timestamp': datetime.now().isoformat()
            })

        price_info = prediction['price_info']
        if price_info['source'] == 'forecasted' and price_info['price'] > 7.0:
            alerts.append({
                'type': 'warning',
                'message': 'High price forecast',
                'detail': f"Ex-farm price forecasted at RM {price_info['price']:.2f} for {prediction['target_date']}",
                'timestamp': datetime.now().isoformat()
            })

        return jsonify({'success': True, 'data': alerts})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health', methods=['GET'])
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'message': 'Chicken Restock Predictor API',
        'status': 'running',
        'endpoints': [
            '/api/predict',
            '/api/predict/week',
            '/api/price/current',
            '/api/price/forecast',
            '/api/price/history',
            '/api/alerts',
            '/health'
        ]
    })
