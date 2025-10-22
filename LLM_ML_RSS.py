import pandas as pd
import numpy as np
import requests
import json
import time
import feedparser
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

TENNIS_RSS_FEEDS = [
    "https://www.atptour.com/rss/news",
    "https://www.wtatennis.com/rss"
]

INITIAL_ELO = 1500
MODEL_NAME = "llama3.2"
API_URL = "http://localhost:11434/api/generate"
MAX_WORKERS = 4
TIMEOUT = 300

class RSSParser:
    def __init__(self):
        self.cache = {}
    
    def fetch_player_news(self, player_name: str, days_back=3):
        if player_name in self.cache:
            return self.cache[player_name]
        
        injury_keywords = ["injur", "withdraw", "retir", "pain", "surgery"]
        recovery_keywords = ["recover", "return", "back", "fit", "heal"]
        player_news = []
        
        for url in TENNIS_RSS_FEEDS:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries:
                    if hasattr(entry, 'published_parsed'):
                        pub_date = datetime(*entry.published_parsed[:6])
                        if (datetime.now() - pub_date).days > days_back:
                            continue
                    
                    content = f"{getattr(entry,'title','')} {getattr(entry,'description','')}".lower()
                    if player_name.lower() in content:
                        news_type = "injury" if any(kw in content for kw in injury_keywords) else "recovery"
                        player_news.append({
                            'title': entry.title,
                            'summary': entry.description,
                            'date': entry.get('published', 'N/A'),
                            'type': news_type
                        })
            except Exception as e:
                print(f"RSS error: {e}")
        
        self.cache[player_name] = player_news
        return player_news

class LLMPredictor:
    def __init__(self, model_name: str, api_url: str):
        self.model_name = model_name
        self.api_url = api_url
    
    def build_prompt(self, match_data: dict, ml_prediction: float):
        home = match_data.get('home_player_name', 'Player 1')
        away = match_data.get('away_player_name', 'Player 2')
        ml_fav = home if ml_prediction > 0.5 else away
        
        prompt = f"""Tennis Match Analysis:

Player 1: {home} (ELO: {match_data.get('elo_home', INITIAL_ELO):.0f})
Player 2: {away} (ELO: {match_data.get('elo_away', INITIAL_ELO):.0f})

Tournament: {match_data.get('tournament_name', 'ATP')}
Surface: {match_data.get('court_surface', 'hard')}
ML Prediction: {ml_fav} ({ml_prediction:.1%})

Respond in JSON format: {{"winner": "NAME", "confidence": 1-10, "reason": "..."}}"""

        if 'injury_report' in match_data and match_data['injury_report']:
            prompt += f"\n\nRecent News:\n{match_data['injury_report']}"
        
        return prompt
    
    def get_prediction(self, prompt: str, retries=3):
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.1}
        }
        
        for attempt in range(retries):
            try:
                response = requests.post(self.api_url, json=payload, timeout=TIMEOUT)
                response.raise_for_status()
                data = response.json()
                
                if 'response' in data:
                    return json.loads(data['response'].strip())
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
        return None

class TennisPredictor:
    def __init__(self):
        self.elo_ratings = {}
        self.player_matches = {}
        self.h2h_history = {}
        self.model = None
        self.features = []
        self.rss_parser = RSSParser()
        self.llm_predictor = LLMPredictor(MODEL_NAME, API_URL)
    
    def load_data(self, past_path: str, future_path: str):
        df_past = pd.read_csv(past_path, low_memory=False)
        df_future = pd.read_csv(future_path, low_memory=False)
        
        df_past['startDate'] = pd.to_datetime(df_past['startDate'], errors='coerce')
        df_future['startDate'] = pd.to_datetime(df_future['startDate'], errors='coerce')
        
        return df_past, df_future
    
    def calculate_elo(self, df: pd.DataFrame):
        if 'winnerCode' in df.columns:
            df['home_win'] = (df['winnerCode'] == 1).astype(int)
        
        df = df.sort_values('startDate').reset_index(drop=True)
        elo_home_list, elo_away_list = [], []
        
        for _, match in df.iterrows():
            home_id = match.get('home_player_id')
            away_id = match.get('away_player_id')
            
            if pd.isna(home_id) or pd.isna(away_id):
                elo_home_list.append(INITIAL_ELO)
                elo_away_list.append(INITIAL_ELO)
                continue
            
            h_matches = self.player_matches.get(home_id, 0)
            a_matches = self.player_matches.get(away_id, 0)
            
            k_h = 40 if h_matches < 30 else 32 if h_matches < 100 else 24
            k_a = 40 if a_matches < 30 else 32 if a_matches < 100 else 24
            
            elo_h = self.elo_ratings.get(home_id, INITIAL_ELO)
            elo_a = self.elo_ratings.get(away_id, INITIAL_ELO)
            
            expected = 1 / (1 + 10 ** ((elo_a - elo_h) / 400))
            actual = match.get('home_win', 0)
            
            elo_home_list.append(elo_h)
            elo_away_list.append(elo_a)
            
            self.elo_ratings[home_id] = elo_h + k_h * (actual - expected)
            self.elo_ratings[away_id] = elo_a + k_a * ((1 - actual) - (1 - expected))
            
            self.player_matches[home_id] = h_matches + 1
            self.player_matches[away_id] = a_matches + 1
            
            if actual == 1:
                self.h2h_history[(home_id, away_id)] = self.h2h_history.get((home_id, away_id), 0) + 1
            elif actual == 0:
                self.h2h_history[(away_id, home_id)] = self.h2h_history.get((away_id, home_id), 0) + 1
        
        df['elo_home'] = elo_home_list
        df['elo_away'] = elo_away_list
        df['elo_diff'] = df['elo_home'] - df['elo_away']
        
        return df
    
    def prepare_features(self, df_past: pd.DataFrame, df_future: pd.DataFrame):
        df_past = self.calculate_elo(df_past)
        
        df_future['elo_home'] = df_future['home_player_id'].map(
            lambda x: self.elo_ratings.get(x, INITIAL_ELO))
        df_future['elo_away'] = df_future['away_player_id'].map(
            lambda x: self.elo_ratings.get(x, INITIAL_ELO))
        df_future['elo_diff'] = df_future['elo_home'] - df_future['elo_away']
        
        base_features = ['elo_diff']
        optional = ['home_player_ranking', 'away_player_ranking', 'court_surface']
        
        self.features = base_features + [f for f in optional 
                                         if f in df_past.columns and f in df_future.columns]
        
        return df_past, df_future
    
    def train_model(self, df_past: pd.DataFrame):
        train_data = df_past.dropna(subset=['home_win'])
        
        if len(train_data) == 0:
            raise ValueError("No training data available")
        
        X = train_data[self.features]
        y = train_data['home_win']
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42)
        
        categorical = [f for f in self.features if train_data[f].dtype == 'object']
        numerical = [f for f in self.features if f not in categorical]
        
        preprocessor = ColumnTransformer(transformers=[
            ('num', SimpleImputer(strategy='median'), numerical),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
        ])
        
        self.model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=150,
                max_depth=5,
                min_samples_split=5,
                random_state=42,
                class_weight='balanced'
            ))
        ])
        
        self.model.fit(X_train, y_train)
        
        try:
            y_proba = self.model.predict_proba(X_val)[:, 1]
            roc = roc_auc_score(y_val, y_proba)
            print(f"Model ROC AUC: {roc:.4f}")
        except Exception:
            print("ROC AUC: N/A")
        
        return self.model
    
    def integrate_rss(self, df: pd.DataFrame):
        for idx, row in df.iterrows():
            home_news = self.rss_parser.fetch_player_news(row['home_player_full_name'])
            away_news = self.rss_parser.fetch_player_news(row['away_player_full_name'])
            
            report = ""
            if home_news:
                report += f"{row['home_player_full_name']}:\n"
                for news in home_news[:2]:
                    status = "⚠️" if news['type'] == "injury" else "✅"
                    report += f"  {status} {news['title']}\n"
            
            if away_news:
                report += f"\n{row['away_player_full_name']}:\n"
                for news in away_news[:2]:
                    status = "⚠️" if news['type'] == "injury" else "✅"
                    report += f"  {status} {news['title']}\n"
            
            df.at[idx, 'injury_report'] = report if report else "No recent news"
        
        return df
    
    def calculate_confidence(self, match_data: dict, ml_prob: float, llm_conf: int):
        base = abs(ml_prob - 0.5) * 200
        
        factors = {
            'elo': abs(match_data.get('elo_diff', 0)) / 200,
            'ranking': abs(match_data.get('home_player_ranking', 100) - 
                          match_data.get('away_player_ranking', 100)) / 50
        }
        
        boost = sum(min(v, 1.0) * 20 for v in factors.values())
        final = (base + boost) * 0.7 + (llm_conf * 10 * 0.3)
        
        return max(30, min(95, final + np.random.uniform(-3, 3)))
    
    def predict_match(self, match_data: dict):
        try:
            ml_input = pd.DataFrame([{f: match_data[f] for f in self.features}])
            ml_prob = self.model.predict_proba(ml_input)[0][1]
        except Exception:
            ml_prob = 0.5
        
        llm_result = self.llm_predictor.get_prediction(
            self.llm_predictor.build_prompt(match_data, ml_prob))
        
        if llm_result and 'winner' in llm_result:
            winner = llm_result['winner']
            llm_conf = min(max(llm_result.get('confidence', 7), 1), 10)
            reason = llm_result.get('reason', '')
        else:
            winner = match_data['home_player_name'] if ml_prob > 0.5 else match_data['away_player_name']
            llm_conf = 7
            reason = f"ML prediction ({ml_prob:.0%})"
        
        confidence = self.calculate_confidence(match_data, ml_prob, llm_conf)
        
        return {
            'winner': winner,
            'confidence': round(confidence, 1),
            'reason': reason,
            'ml_probability': round(ml_prob * 100, 2)
        }

def run_predictions(past_csv: str, future_csv: str, output_file: str, use_rss=False):
    predictor = TennisPredictor()
    
    print("Loading data...")
    df_past, df_future = predictor.load_data(past_csv, future_csv)
    
    print("Engineering features...")
    df_past, df_future = predictor.prepare_features(df_past, df_future)
    
    print("Training model...")
    predictor.train_model(df_past)
    
    if use_rss:
        print("Integrating RSS data...")
        df_future = predictor.integrate_rss(df_future)
    
    print(f"Generating predictions for {len(df_future)} matches...")
    results = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(predictor.predict_match, row.to_dict()): idx 
                  for idx, row in df_future.iterrows()}
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            prediction = future.result()
            if prediction:
                results.append(prediction)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({'predictions': results}, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(results)} predictions to {output_file}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--past', required=True)
    parser.add_argument('--future', required=True)
    parser.add_argument('--output', default='predictions.json')
    parser.add_argument('--use-rss', action='store_true')
    args = parser.parse_args()
    
    run_predictions(args.past, args.future, args.output, args.use_rss)
