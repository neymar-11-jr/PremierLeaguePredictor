import requests
import pandas as pd

def fetch_live_fixtures(league_code="PL", season="2025"):
    """
    从 football-data.org 获取指定联赛和赛季的赛程数据。
    """
    API_TOKEN = "bc668c08aea543c7a2c238ac6e1842a5" # 你的Token
    headers = {"X-Auth-Token": API_TOKEN}
    url = f"https://api.football-data.org/v4/competitions/{league_code}/matches?season={season}"
    
    print(f"正在从API获取 {league_code} {season} 赛季数据...")
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        matches = response.json()["matches"]
        rows = []
        for m in matches:
            if m["status"] == "FINISHED": # 只获取已完成的比赛
                rows.append({
                    "date": m["utcDate"][:10],
                    "home_team": m["homeTeam"]["name"],
                    "away_team": m["awayTeam"]["name"],
                    "home_score": m["score"]["fullTime"]["home"],
                    "away_score": m["score"]["fullTime"]["away"],
                })
        df = pd.DataFrame(rows)
        print(f"✅ 成功获取 {len(df)} 场 {season} 赛季比赛数据。")
        return df
    else:
        print(f"❌ 请求失败，状态码：{response.status_code}")
        return None