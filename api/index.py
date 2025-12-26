from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List
import json
import os
import sys

app = Flask(__name__)
CORS(app)

TMP_HISTORY_PATH = "/tmp/prediction_history.json"
TMP_CSV_PATH = "/tmp/ExFarmPrice.csv"

print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Files in directory: {os.listdir('.')}")

def ensure_tmp_csv_exists():
    """Copy CSV from project root to /tmp on cold start"""
    if not os.path.exists(TMP_CSV_PATH):
        try:
            possible_paths = [
                "ExFarmPrice.csv",
                "../ExFarmPrice.csv",
                "./ExFarmPrice.csv",
                os.path.join(os.path.dirname(__file__), "ExFarmPrice.csv"),
                os.path.join(os.path.dirname(__file__), "..", "ExFarmPrice.csv")
            ]
            
            csv_found = False
            for src_path in possible_paths:
                print(f"Looking for CSV at: {src_path}")
                if os.path.exists(src_path):
                    print(f"Found CSV at: {src_path}")
                    with open(src_path, "rb") as fr:
                        data = fr.read()
                    with open(TMP_CSV_PATH, "wb") as fw:
                        fw.write(data)
                    csv_found = True
                    print(f"CSV copied from {src_path} to {TMP_CSV_PATH}")
                    break
            
            if not csv_found:
                print("Creating default CSV template...")
                data = """Date_Range,Avg_Price
01.01.2024 - 07.01.2024,6.50
08.01.2024 - 14.01.2024,6.45
15.01.2024 - 21.01.2024,6.60
22.01.2024 - 28.01.2024,6.55
29.01.2024 - 04.02.2024,6.70
05.02.2024 - 11.02.2024,6.65
12.02.2024 - 18.02.2024,6.75
19.02.2024 - 25.02.2024,6.80
26.02.2024 - 03.03.2024,6.70
04.03.2024 - 10.03.2024,6.65"""
                with open(TMP_CSV_PATH, "w") as f:
                    f.write(data)
                print("Created default CSV template")
        except Exception as e:
            print(f"Error preparing tmp CSV: {e}")
            with open(TMP_CSV_PATH, "w") as f:
                f.write("Date_Range,Avg_Price\n01.01.2024 - 07.01.2024,6.50")

ensure_tmp_csv_exists()

class LiveCalendarService:
    """Service to fetch Malaysian holidays from Calendarific API"""
    
    def __init__(self, api_key: str = None):
        self.calendarific_api_key = api_key
        self.cache = {}
        self.cache_expiry = {}
        
    def get_malaysian_holidays(self, year: int) -> List[Dict]:
        """Fetch Malaysian public holidays from Calendarific API"""
        cache_key = f"holidays_{year}"
        
        if cache_key in self.cache:
            if datetime.now() < self.cache_expiry.get(cache_key, datetime.now()):
                print(f"Using cached holidays for {year}")
                return self.cache[cache_key]
        
        if not self.calendarific_api_key:
            print("No Calendarific API key provided, cannot fetch holidays")
            return []
        
        try:
            url = "https://calendarific.com/api/v2/holidays"
            params = {
                'api_key': self.calendarific_api_key,
                'country': 'MY',
                'year': year,
                'type': 'national,local'
            }
            
            print(f"Fetching holidays from Calendarific for year {year}...")
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if response.status_code == 200 and data.get('meta', {}).get('code') == 200:
                holidays = []
                for holiday in data.get('response', {}).get('holidays', []):
                    holidays.append({
                        'name': holiday['name'],
                        'date': holiday['date']['iso'],
                        'type': holiday.get('type', ['national'])[0],
                        'description': holiday.get('description', '')
                    })
                
                self.cache[cache_key] = holidays
                self.cache_expiry[cache_key] = datetime.now() + timedelta(hours=24)
                print(f"Successfully fetched {len(holidays)} holidays for {year}")
                return holidays
            else:
                print(f"Calendarific API error: {data}")
                return []
                
        except Exception as e:
            print(f"Error fetching holidays from API: {e}")
            return []
    
    def get_event_factor(self, holiday_name: str, holiday_type: str) -> Dict:
        """Determine demand factor based on holiday type and name"""
        
        major_festivals = {
            'Chinese New Year': {'factor': 0.40, 'pre_days': 5, 'post_days': 2},
            'Hari Raya Aidilfitri': {'factor': 0.50, 'pre_days': 5, 'post_days': 2},
            'Hari Raya Haji': {'factor': 0.30, 'pre_days': 3, 'post_days': 1},
            'Hari Raya Aidiladha': {'factor': 0.30, 'pre_days': 3, 'post_days': 1},
            'Christmas': {'factor': 0.30, 'pre_days': 3, 'post_days': 1},
            'Deepavali': {'factor': 0.25, 'pre_days': 3, 'post_days': 1},
            'Diwali': {'factor': 0.25, 'pre_days': 3, 'post_days': 1},
        }
        
        for festival_key, festival_data in major_festivals.items():
            if festival_key.lower() in holiday_name.lower():
                return festival_data
        
        if holiday_type == 'national':
            return {'factor': 0.20, 'pre_days': 2, 'post_days': 0}
        else:
            return {'factor': 0.15, 'pre_days': 1, 'post_days': 0}
    
    def get_calendar_events(self, target_date: datetime) -> Dict:
        """Get calendar events for a specific date from live API"""
        
        if not self.calendarific_api_key:
            print("No Calendarific API key configured, returning normal day")
            return {
                'has_event': False,
                'event_name': 'Normal day (no calendar data)',
                'factor': 0.0,
                'type': 'normal',
                'source': 'no_api_key'
            }
        
        try:
            current_year = datetime.now().year
            target_year = target_date.year
            
            holidays = []
            if target_year == current_year:
                holidays.extend(self.get_malaysian_holidays(current_year))
            if target_year == current_year + 1:
                holidays.extend(self.get_malaysian_holidays(current_year + 1))
            
            if not holidays:
                print(f"No holiday data available for {target_date}")
                return self._get_rule_based_events(target_date)
            
            result = self._process_holidays(target_date, holidays, 'live_api')
            if result['has_event']:
                return result
            else:
                return self._get_rule_based_events(target_date)
                
        except Exception as e:
            print(f"Error in live calendar fetch: {e}")
            return self._get_rule_based_events(target_date)
    
    def _process_holidays(self, target_date: datetime, holidays: List[Dict], source: str) -> Dict:
        """Process holidays list and determine if target date has events"""
        
        target_date_str = target_date.strftime('%Y-%m-%d')
        
        for holiday in holidays:
            if holiday['date'] == target_date_str:
                event_config = self.get_event_factor(holiday['name'], holiday['type'])
                return {
                    'has_event': True,
                    'event_name': holiday['name'],
                    'factor': event_config['factor'],
                    'type': 'festival',
                    'source': source
                }
        
        for holiday in holidays:
            holiday_date = datetime.fromisoformat(holiday['date'])
            days_before = (holiday_date - target_date).days
            
            event_config = self.get_event_factor(holiday['name'], holiday['type'])
            
            if 0 < days_before <= event_config['pre_days']:
                proximity_factor = event_config['factor'] * (1 - days_before / event_config['pre_days']) * 0.7
                return {
                    'has_event': True,
                    'event_name': f"{days_before} days before {holiday['name']}",
                    'factor': proximity_factor,
                    'type': 'pre-festival',
                    'source': source
                }
            
            days_after = (target_date - holiday_date).days
            if 0 < days_after <= event_config['post_days']:
                return {
                    'has_event': True,
                    'event_name': f"{days_after} days after {holiday['name']}",
                    'factor': -0.25,
                    'type': 'post-festival',
                    'source': source
                }
        
        return {'has_event': False}
    
    def _get_rule_based_events(self, target_date: datetime) -> Dict:
        """Get rule-based events when no holiday data is available"""
        
        ramadan_info = self._check_ramadan_period(target_date)
        if ramadan_info['has_event']:
            return ramadan_info
        
        school_holiday_info = self._check_school_holidays(target_date)
        if school_holiday_info['has_event']:
            return school_holiday_info
        
        if target_date.weekday() == 4:
            return {
                'has_event': True,
                'event_name': 'Friday (weekend preparation)',
                'factor': 0.12,
                'type': 'friday',
                'source': 'rule_based'
            }
        
        return {
            'has_event': False,
            'event_name': 'Normal day',
            'factor': 0.0,
            'type': 'normal',
            'source': 'rule_based'
        }
    
    def _check_ramadan_period(self, target_date: datetime) -> Dict:
        """Check if date falls during Ramadan (approximate dates)"""
        ramadan_periods = {
            2025: {'start': datetime(2025, 3, 1), 'end': datetime(2025, 3, 30)},
            2026: {'start': datetime(2026, 2, 18), 'end': datetime(2026, 3, 19)},
            2027: {'start': datetime(2027, 2, 8), 'end': datetime(2027, 3, 9)},
        }
        
        year = target_date.year
        if year in ramadan_periods:
            period = ramadan_periods[year]
            if period['start'] <= target_date <= period['end']:
                days_to_end = (period['end'] - target_date).days
                if days_to_end <= 14:
                    ramadan_factor = 0.15 + (14 - days_to_end) / 14 * 0.20
                    return {
                        'has_event': True,
                        'event_name': 'Ramadan (approaching Raya)',
                        'factor': ramadan_factor,
                        'type': 'ramadan',
                        'source': 'calculated'
                    }
                else:
                    return {
                        'has_event': True,
                        'event_name': 'Ramadan',
                        'factor': 0.10,
                        'type': 'ramadan',
                        'source': 'calculated'
                    }
        
        return {'has_event': False}
    
    def _check_school_holidays(self, target_date: datetime) -> Dict:
        """Check approximate school holiday periods"""
        school_holidays = [
            {'start': datetime(2024, 11, 23), 'end': datetime(2025, 1, 5), 'name': 'Year End', 'factor': 0.15},
            {'start': datetime(2025, 3, 22), 'end': datetime(2025, 3, 30), 'name': 'Mid Year', 'factor': 0.12},
            {'start': datetime(2025, 5, 24), 'end': datetime(2025, 6, 8), 'name': 'Mid Year', 'factor': 0.12},
            {'start': datetime(2025, 8, 16), 'end': datetime(2025, 8, 24), 'name': 'Short Break', 'factor': 0.10},
            {'start': datetime(2025, 11, 22), 'end': datetime(2026, 1, 4), 'name': 'Year End', 'factor': 0.15},
            {'start': datetime(2026, 3, 28), 'end': datetime(2026, 4, 5), 'name': 'Mid Year', 'factor': 0.12},
            {'start': datetime(2026, 5, 23), 'end': datetime(2026, 6, 7), 'name': 'Mid Year', 'factor': 0.12},
            {'start': datetime(2026, 8, 22), 'end': datetime(2026, 8, 30), 'name': 'Short Break', 'factor': 0.10},
            {'start': datetime(2026, 11, 21), 'end': datetime(2027, 1, 3), 'name': 'Year End', 'factor': 0.15},
        ]
        
        for holiday in school_holidays:
            if holiday['start'] <= target_date <= holiday['end']:
                return {
                    'has_event': True,
                    'event_name': f"School Holiday ({holiday['name']})",
                    'factor': holiday['factor'],
                    'type': 'school-holiday',
                    'source': 'configured'
                }
        
        return {'has_event': False}


class ChickenRestockPredictor:
    def __init__(self, api_url: str):
        self.api_url = api_url
        self.RESTOCK_DAYS = [0, 3, 5]
        self.history_file = TMP_HISTORY_PATH
        
        # Demand level thresholds based on total adjustment factor
        self.DEMAND_THRESHOLDS = {
            'low': -0.15,      # total_adjustment < -0.15
            'medium_low': 0.0, # -0.15 <= total_adjustment < 0.0
            'medium': 0.15,    # 0.0 <= total_adjustment < 0.15
            'medium_high': 0.30, # 0.15 <= total_adjustment < 0.30
            'high': 0.30       # total_adjustment >= 0.30
        }
        
        calendarific_key = os.getenv('CALENDARIFIC_API_KEY', '')
        self.calendar_service = LiveCalendarService(calendarific_key)
        
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

    def add_historical_record(self, prediction: Dict, actual: str = None):
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

        correct_predictions = sum(1 for h in recent_history 
                                 if h['prediction']['demand_level'] == h['actual'])
        
        total_predictions = len(recent_history)
        accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0

        return {
            'accuracy_percentage': round(accuracy, 2),
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions,
            'period_days': days
        }

    def fetch_current_stock(self) -> Dict:
        try:
            response = requests.get(self.api_url, timeout=10)
            data = response.json()

            if data.get('success', False):
                factory_stock = 0
                kiosk_stock = 0

                for item in data.get('factory_data', []):
                    if item['item_id'] == 11:
                        factory_stock = int(item['stock_count'])
                        break

                for kiosk in data.get('kiosk_data', []):
                    for item in kiosk.get('items', []):
                        if item['item_id'] == 11:
                            kiosk_stock += int(item['stock_count'])

                return {
                    'factory_stock': factory_stock,
                    'kiosk_stock': kiosk_stock
                }
            else:
                raise Exception("API returned unsuccessful response")

        except Exception as e:
            print(f"Error fetching stock data: {e}")
            return {'factory_stock': 500, 'kiosk_stock': 300}

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
            if len(df) == 0:
                return 6.5
                
            df['end_date'] = df['Date_Range'].apply(self.parse_date_range)
            df = df.dropna(subset=['end_date'])
            if len(df) == 0:
                return 6.5
                
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
            if len(df) == 0:
                return {
                    'forecasted_price': 6.5,
                    'confidence': 'Low',
                    'method': 'fallback',
                    'factors': {}
                }

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
        """Get calendar events using live calendar service"""
        return self.calendar_service.get_calendar_events(target_date)

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

    def determine_demand_level(self, total_adjustment: float) -> Dict:
        """Determine demand level based on total adjustment factor"""
        
        if total_adjustment >= self.DEMAND_THRESHOLDS['high']:
            level = "High"
            description = "Significantly elevated demand expected"
            recommendation = "Prepare maximum stock. Consider ordering extra supplies."
        elif total_adjustment >= self.DEMAND_THRESHOLDS['medium_high']:
            level = "Medium-High"
            description = "Above average demand anticipated"
            recommendation = "Stock above normal levels to meet increased demand."
        elif total_adjustment >= self.DEMAND_THRESHOLDS['medium']:
            level = "Medium"
            description = "Normal to slightly elevated demand"
            recommendation = "Maintain standard stock levels with slight buffer."
        elif total_adjustment >= self.DEMAND_THRESHOLDS['medium_low']:
            level = "Medium-Low"
            description = "Normal to slightly below average demand"
            recommendation = "Standard stock levels appropriate."
        else:
            level = "Low"
            description = "Below average demand expected"
            recommendation = "Reduce stock levels to avoid excess inventory."
        
        return {
            'level': level,
            'description': description,
            'recommendation': recommendation,
            'adjustment_factor': round(total_adjustment, 3)
        }

    def predict_restock_demand(self, target_date: datetime = None) -> Dict:
        """Predict demand level instead of quantity"""
        
        if target_date is None:
            today = datetime.now()
            for i in range(7):
                check_date = today + timedelta(days=i)
                if check_date.weekday() in self.RESTOCK_DAYS:
                    target_date = check_date
                    break
            else:
                target_date = today + timedelta(days=1)

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
        calendar_factor = calendar_info.get('factor', 0.0)
        day_factor = self.calculate_day_of_week_factor(target_date)

        total_adjustment = inventory_factor + price_factor + calendar_factor + day_factor
        
        # Determine demand level
        demand_info = self.determine_demand_level(total_adjustment)

        result = {
            'target_date': target_date.strftime('%Y-%m-%d (%A)'),
            'demand_level': demand_info['level'],
            'demand_description': demand_info['description'],
            'recommendation': demand_info['recommendation'],
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
                result = self.predict_restock_demand(check_date)
                predictions.append(result)

        return predictions

    def get_price_history(self, days: int = 90) -> Dict:
        try:
            df = pd.read_csv(TMP_CSV_PATH)
            if len(df) == 0:
                return {'success': False, 'error': 'No data available'}

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
    api_url="https://inventory.ayamgorengkimiez.my/api/analytics/96e27e560a23a5a21978005c3d69add802bfa5b9be3cb6c1f7735e51db80bfe2/overview"
)

# API Routes
@app.route('/api/predict', methods=['GET'])
def get_prediction():
    try:
        result = predictor.predict_restock_demand()
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
        actual_demand = data.get('actual_demand')  # Should be "Low", "Medium-Low", "Medium", "Medium-High", or "High"

        if not date or not actual_demand:
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400

        valid_levels = ["Low", "Medium-Low", "Medium", "Medium-High", "High"]
        if actual_demand not in valid_levels:
            return jsonify({'success': False, 'error': f'Invalid demand level. Must be one of: {valid_levels}'}), 400

        target_date = datetime.strptime(date, '%Y-%m-%d')
        prediction = predictor.predict_restock_demand(target_date)
        predictor.add_historical_record(prediction, actual_demand)

        return jsonify({'success': True, 'message': 'Actual demand level recorded'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    try:
        prediction = predictor.predict_restock_demand()
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

        if prediction['demand_level'] in ['High', 'Medium-High']:
            alerts.append({
                'type': 'info',
                'message': f'{prediction['demand_level']} demand period approaching',
                'detail': f"{prediction['recommendation']} Event: {prediction['calendar_event']['event_name']}",
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

@app.route('/api/calendar/test', methods=['GET'])
def test_calendar():
    """Test endpoint to check calendar integration"""
    try:
        date_str = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
        target_date = datetime.strptime(date_str, '%Y-%m-%d')
        
        event_info = predictor.get_calendar_events(target_date)
        
        return jsonify({
            'success': True,
            'date': date_str,
            'event_info': event_info,
            'api_status': 'working' if event_info.get('source') == 'live_api' else 'rule_based',
            'has_api_key': bool(os.getenv('CALENDARIFIC_API_KEY'))
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/calendar/holidays', methods=['GET'])
def get_holidays():
    """Get all holidays for a specific year"""
    try:
        year = int(request.args.get('year', datetime.now().year))
        holidays = predictor.calendar_service.get_malaysian_holidays(year)
        
        return jsonify({
            'success': True,
            'year': year,
            'count': len(holidays),
            'holidays': holidays,
            'source': 'live_api' if holidays else 'no_data'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health', methods=['GET'])
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'calendar_api_configured': bool(os.getenv('CALENDARIFIC_API_KEY'))
    })

@app.route('/debug', methods=['GET'])
def debug_info():
    """Debug endpoint to check system status"""
    return jsonify({
        'status': 'running',
        'python_version': sys.version,
        'current_time': datetime.now().isoformat(),
        'csv_exists': os.path.exists(TMP_CSV_PATH),
        'csv_size': os.path.getsize(TMP_CSV_PATH) if os.path.exists(TMP_CSV_PATH) else 0,
        'tmp_files': os.listdir('/tmp') if os.path.exists('/tmp') else [],
        'working_dir': os.getcwd(),
        'files_in_dir': os.listdir('.')
    })

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'message': 'Chicken Restock Demand Predictor API',
        'status': 'running',
        'version': '2.0 - Demand Levels',
        'calendar_integration': 'live_api_only' if os.getenv('CALENDARIFIC_API_KEY') else 'no_api_key',
        'demand_levels': ['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High'],
        'endpoints': [
            '/api/predict - Get demand prediction',
            '/api/predict/week - Get weekly predictions',
            '/api/price/current - Current price',
            '/api/price/forecast - Price forecast',
            '/api/price/history - Price history',
            '/api/alerts - Get alerts',
            '/api/record - Record actual demand',
            '/api/accuracy - Prediction accuracy',
            '/api/calendar/test - Test calendar',
            '/api/calendar/holidays - Get holidays',
            '/health - Health check',
            '/debug - Debug info'
        ]
    })

