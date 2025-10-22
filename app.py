import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import pandas as pd
from pandas.errors import EmptyDataError
from fastapi import FastAPI, HTTPException, Query, Path as PathParam
from fastapi.responses import JSONResponse
from scripts.update_memory import update_memory
from scripts.update_player_stats_atp import update_player_stats as update_player_stats_atp
from scripts.update_player_stats_wta import update_player_stats as update_player_stats_wta
import pipeline_wta
import test_on_future_games_wta as test_wta

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
HISTORY_DIR = BASE_DIR / "history"
DATA_DIR.mkdir(exist_ok=True)
HISTORY_DIR.mkdir(exist_ok=True)

def filter_predictions_by_date(predictions, date_str):
    
    if not predictions:
        return []
    
    filtered = []
    for pred in predictions:
        
        if 'startDate' in pred and pred['startDate'].startswith(date_str):
            filtered.append(pred)
        elif 'startDateTime' in pred and pred['startDateTime'].startswith(date_str):
            filtered.append(pred)
    return filtered

def load_past_games_for_date(date_obj: datetime, league: str):
    csv_path = Path("datasets") / f"tennis_matches_past_{league}.csv"
    if not csv_path.exists(): return []
    df = pd.read_csv(csv_path, low_memory=False)
    date_col = df.columns[-2]; time_col = df.columns[-1]
    df['startDate'] = pd.to_datetime(df[date_col], errors='coerce')
    df['startTime'] = pd.to_datetime(df[time_col], format='%H:%M', errors='coerce').dt.time
    df = df.dropna(subset=['startDate'])
    df = df[df['startDate'].dt.date == date_obj.date()]
    if 'tennis_match_slug' in df.columns:
        df = df.drop_duplicates(subset=['tennis_match_slug'])
    games=[]
    for _, r in df.iterrows():
        dt = r['startDate']
        if pd.notna(r['startTime']): dt = datetime.combine(r['startDate'].date(), r['startTime'])
        games.append({
            "tennis_match_slug": r.get("tennis_match_slug",""),
            "tournament": r.get("tournament_slug",""),
            "home_player": r.get("home_player_name",""),
            "away_player": r.get("away_player_name",""),
            "startDateTime": dt.isoformat(),
            "result": r.get("winner_player_slug","")
        })
    return games

@app.get("/run")
async def run_pipeline(
    date: str = Query(..., description="YYYY-MM-DD Europe/Berlin"),
    league: str = Query(..., description="ATP or WTA")
):
    league_u = league.upper()
    if league_u not in ("ATP","WTA"): raise HTTPException(400,"league має бути 'ATP' або 'WTA'")
    league_n = league_u.lower()
    try: date_obj = datetime.strptime(date, "%Y-%m-%d")
    except ValueError: raise HTTPException(400,"Invalid date format")
    
    
    update_memory(date, league_n)
    
    
    past_csv = BASE_DIR / "datasets" / f"tennis_matches_past_{league_n}.csv"
    future_csv = BASE_DIR / "datasets" / f"tennis_matches_future_{league_n}.csv"
    output_json = DATA_DIR / f"predictions_{league_n}.json"
    
    if league_n == 'atp':
        script = BASE_DIR / "LLM_ML_RSS.py"
        
        try:
            
            subprocess.run(["python", str(script),
                            "--past", str(past_csv),
                            "--future", str(future_csv),
                            "--output", str(output_json)], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running ATP script: {e}")
            raise HTTPException(500,"Error generating ATP predictions")
        
        try:
            
            if output_json.exists():
                data = json.loads(output_json.read_text(encoding='utf-8'))
                all_preds = data.get('predictions', [])
                
                
                date_preds = filter_predictions_by_date(all_preds, date)
                
                
                with open(output_json, 'w', encoding='utf-8') as f:
                    json.dump({"predictions": date_preds}, f, ensure_ascii=False, indent=2)
                
                preds = {"predictions": date_preds}
            else:
                preds = {"predictions": []}
        except Exception as e:
            logger.warning(f"Failed processing ATP JSON: {e}")
            preds = {"predictions": []}
            
    else:  
        script = BASE_DIR / "LLM_ML_RSS.new.py" 
        
        try:
            
            subprocess.run(["python", str(script),
                            "--past", str(past_csv),
                            "--future", str(future_csv),
                            "--output", str(output_json)], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running WTA script: {e}")
            raise HTTPException(500,"Error generating WTA predictions")
        
        try:
            
            if output_json.exists():
                data = json.loads(output_json.read_text(encoding='utf-8'))
                all_preds = data.get('predictions', [])
                
                
                date_preds = filter_predictions_by_date(all_preds, date)
                
                
                with open(output_json, 'w', encoding='utf-8') as f:
                    json.dump({"predictions": date_preds}, f, ensure_ascii=False, indent=2)
                
                preds = {"predictions": date_preds}
            else:
                preds = {"predictions": []}
        except Exception as e:
            logger.warning(f"Failed processing WTA JSON: {e}")
            preds = {"predictions": []}
    
    
    history_json = HISTORY_DIR / f"predictions_{league_n}_history.json"
    new_records = preds.get("predictions", preds)
    
   
    date_new_records = filter_predictions_by_date(new_records, date)
    
    existing = []
    if history_json.exists():
        try:
            content = json.loads(history_json.read_text(encoding='utf-8'))
            existing = content.get('history', []) if isinstance(content, dict) else content
        except Exception:
            existing = []

   
    combined = {
        (item.get('tennis_match_slug') or item.get('match_slug')): item
        for item in existing
        if item.get('tennis_match_slug') or item.get('match_slug')
    }
    
    
    for rec in date_new_records:
        key = rec.get('tennis_match_slug') or rec.get('match_slug')
        if key:
            combined[key] = rec
    
    
    records = sorted(combined.values(), key=lambda x: x.get('tennis_match_slug') or x.get('match_slug'))[-1000:]
    
   
    with open(history_json, 'w', encoding='utf-8') as f:
        json.dump({"history": records}, f, ensure_ascii=False, indent=2)

    
    past_games = load_past_games_for_date(date_obj, league_n)
    
    return {
        "date": date,
        "league": league_u,
        "future_predictions": {"predictions": date_new_records},
        "past_games": past_games
    }


@app.get("/predictions/{league}")
async def get_predictions(
    league: str = PathParam(..., description="Ліга: ATP або WTA")
):
    league_upper = league.upper()
    if league_upper not in ("ATP", "WTA"):
        raise HTTPException(status_code=400, detail="league має бути 'ATP' або 'WTA'")
    league_norm = league_upper.lower()

    json_path = DATA_DIR / f"predictions_{league_norm}.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="Prediction file not found")
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {e}")
    return JSONResponse(content=data)


@app.get("/predictions/{league}/history")
async def get_prediction_history(
    league: str = PathParam(..., description="Ліга: ATP або WTA")
):
    league_upper = league.upper()
    if league_upper not in ("ATP", "WTA"):
        raise HTTPException(status_code=400, detail="league має бути 'ATP' або 'WTA'")
    league_norm = league_upper.lower()

    history_json = HISTORY_DIR / f"predictions_{league_norm}_history.json"
    if not history_json.exists():
        raise HTTPException(status_code=404, detail="History file not found")

    try:
        content = json.loads(history_json.read_text(encoding='utf-8'))
        items = content.get("history") if isinstance(content, dict) else content
        history = items[-50:]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading history file: {e}")

    return JSONResponse(content={"history": history})


@app.post("/refresh/{league}")
async def refresh_predictions(
    league: str = PathParam(..., description="Ліга: ATP або WTA")
):
    league_upper = league.upper()
    if league_upper not in ("ATP", "WTA"):
        raise HTTPException(status_code=400, detail="league має бути 'ATP' або 'WTA'")
    league_norm = league_upper.lower()

    logger.info(f"Starting refresh for league={league_upper}")
    current_date = datetime.now().strftime("%Y-%m-%d")

    try:
        update_memory(current_date, league_norm)
        if league_norm == "atp":
            update_player_stats_atp()
            preds = test_atp.predict_future_games()
        else:
            update_player_stats_wta()
            preds = test_wta.predict_future_games()
    except EmptyDataError:
        logger.warning("Future matches CSV is empty or not found; no new predictions generated")
        preds = {"predictions": []}
    except Exception as e:
        logger.error(f"Error during refresh: {e}")
        raise HTTPException(status_code=500, detail="Error generating fresh predictions")

    
    live_json = f"predictions_{league_norm}_live.json"
    with open(DATA_DIR / live_json, "w", encoding="utf-8") as lf:
        json.dump({"predictions": preds.get("predictions", preds)}, lf, ensure_ascii=False, indent=2)

    rows = []
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    for p in preds.get("predictions", []):
        winner = p.get("winner_player_slug") or p.get("predicted_winner")
        confidence = p.get("confidence_pct") or p.get("probability")
        rows.append({"timestamp": ts, "winner": winner, "confidence_pct": confidence})
    df = pd.DataFrame(rows)
    csv_path = DATA_DIR / f"{league_norm}_live_stats.csv"
    df.to_csv(csv_path, mode="a", index=False, header=not csv_path.exists(), encoding="utf-8")

    return {"league": league_upper, "updated_predictions": preds}


def load_predictions(league: str) -> pd.DataFrame:
    fn = DATA_DIR / f"predictions_{league}.json"
    if not fn.exists():
        return pd.DataFrame()
    data = json.loads(fn.read_text(encoding="utf-8"))
    df = pd.DataFrame(data.get("predictions", []))
    
    if 'tennis_match_slug' in df.columns:
        df['match_slug'] = df['tennis_match_slug']
    if 'winner_player_slug' in df.columns:
        df['predicted_winner'] = df['winner_player_slug']
    
    if 'confidence_pct' in df.columns:
        df['confidence_pct'] = df['confidence_pct']
    elif 'probability' in df.columns:
        df['confidence_pct'] = df['probability']
    
    return df[['match_slug', 'predicted_winner', 'confidence_pct']]


@app.get("/report/{league}")
async def report(
    league: str = PathParam(..., description="Ліга: ATP або WTA")
):
    league_upper = league.upper()
    if league_upper not in ("ATP", "WTA"):
        raise HTTPException(status_code=400, detail="league має бути 'ATP' або 'WTA'")
    league_norm = league_upper.lower()
    report_date = datetime.now().date()
    date_str = report_date.strftime("%Y-%m-%d")
    try:
        update_memory(date_str, league_norm)
    except Exception as e:
        logger.error(f"Error in update_memory: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating memory: {e}")
    df_pred = load_predictions(league_norm)
    if df_pred.empty:
        return {"message": "No predictions for today", "matches_evaluated": 0, "accuracy_pct": 0}
    past_games = load_past_games_for_date(datetime.combine(report_date, datetime.min.time()), league_norm)
    if not past_games:
        return {"message": "No real results for today", "matches_evaluated": 0, "accuracy_pct": 0}
    df_act = pd.DataFrame(past_games)
    df_act = df_act.rename(columns={'result': 'actual_winner', 'tennis_match_slug': 'match_slug'})
    df_act['match_slug'] = df_act['match_slug'].astype(str)
    df = pd.merge(df_pred, df_act[['match_slug', 'actual_winner']], on='match_slug', how='inner')
    if df.empty:
        return {"message": "No matches found for comparison", "matches_evaluated": 0, "accuracy_pct": 0}
    df['correct'] = df['predicted_winner'] == df['actual_winner']
    accuracy = df['correct'].mean() * 100
    
    df = df.merge(
        df_act[['match_slug', 'tournament', 'home_player', 'away_player', 'startDateTime']],
        on='match_slug', how='left'
    )
    
    if 'confidence_pct' not in df.columns and 'confidence' in df.columns:
        df['confidence_pct'] = df['confidence'] * 100
    elif 'confidence_pct' not in df.columns:
        df['confidence_pct'] = None  
    
    df = df[[
        'match_slug',
        'predicted_winner',
        'actual_winner',
        'correct',
        'confidence_pct',
        'tournament',
        'home_player',
        'away_player',
        'startDateTime'
    ]]
    df.fillna("", inplace=True)
   
    out_fn = HISTORY_DIR / f"daily_report_{league_norm}.csv"
    df.to_csv(out_fn, index=False, encoding='utf-8')
    logger.info(f"Daily report saved to {out_fn}")
    
    report_records = df.to_dict(orient='records')
    return {
        "date": date_str,
        "league": league_upper,
        "matches_evaluated": len(df),
        "confidence_pct": round(accuracy, 2),
        "report": report_records
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
