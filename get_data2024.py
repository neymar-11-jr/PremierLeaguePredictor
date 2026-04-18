import requests
import pandas as pd

API_TOKEN = "bc668c08aea543c7a2c238ac6e1842a5"
headers = {"X-Auth-Token": API_TOKEN}

# 请求英超2024赛季完赛数据
url = "https://api.football-data.org/v4/competitions/PL/matches?season=2024"
response = requests.get(url, headers=headers)

if response.status_code == 200:
    matches = response.json()["matches"]
    rows = []
    for m in matches:
        if m["status"] == "FINISHED":
            rows.append({
                "date": m["utcDate"][:10],
                "home_team": m["homeTeam"]["name"],
                "away_team": m["awayTeam"]["name"],
                "home_score": m["score"]["fullTime"]["home"],
                "away_score": m["score"]["fullTime"]["away"],
            })
    df = pd.DataFrame(rows)
    df.to_csv("premier_league_2024.csv", index=False)
    print(f"✅ 成功！已获取 {len(df)} 场比赛数据。")
    print("文件已保存为：premier_league_2023.csv")
else:
    print(f"❌ 请求失败，状态码：{response.status_code}")