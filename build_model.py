# ========== 导入工具库 ==========
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data_fetcher import fetch_live_fixtures
import joblib

# ========== 任务一：合并三个赛季数据 ==========
# 1. 先加载本地历史数据
df23 = pd.read_csv("premier_league_2023.csv")
df24 = pd.read_csv("premier_league_2024.csv")
df25 = pd.read_csv("premier_league_2025.csv")
df = pd.concat([df23, df24, df25], ignore_index=True)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

print(f"✅ 合并完成！共 {len(df)} 场比赛，已保存为 all_seasons.csv")
df.to_csv("all_seasons.csv", index=False)

# ========== 任务二：构造特征 ==========

def calc_team_form(team, current_date, df, N=5):
    """计算某队在指定日期前的近N场胜率"""
    home_games = df[(df["home_team"] == team) & (df["date"] < current_date)].copy()
    away_games = df[(df["away_team"] == team) & (df["date"] < current_date)].copy()
    home_games["win"] = home_games["home_score"] > home_games["away_score"]
    away_games["win"] = away_games["away_score"] > away_games["home_score"]
    all_games = pd.concat([home_games, away_games]).sort_values("date", ascending=False)
    last_N = all_games.head(N)
    if len(last_N) == 0:
        # 无历史数据时返回联赛整体主胜率作为先验
        return (df["home_score"] > df["away_score"]).mean()
    return last_N["win"].sum() / len(last_N)

def calc_h2h_advantage(home, away, current_date, df, N=3):
    """计算主队面对客队的近期交锋胜率"""
    h2h = df[
        ((df["home_team"] == home) & (df["away_team"] == away)) |
        ((df["home_team"] == away) & (df["away_team"] == home))
    ]
    h2h = h2h[h2h["date"] < current_date].sort_values("date", ascending=False).head(N)
    if len(h2h) == 0:
        return 0.5
    home_wins = 0
    for _, row in h2h.iterrows():
        if row["home_team"] == home:
            if row["home_score"] > row["away_score"]:
                home_wins += 1
        else:
            if row["away_score"] > row["home_score"]:
                home_wins += 1
    return home_wins / len(h2h)

def calc_rolling_avg(team, current_date, df, stat_col, is_home, N=5):
    """计算某队近N场某项统计的平均值（如进球、失球）"""
    if is_home:
        games = df[(df["home_team"] == team) & (df["date"] < current_date)].copy()
        values = games[stat_col]
    else:
        games = df[(df["away_team"] == team) & (df["date"] < current_date)].copy()
        values = games[stat_col]
    if len(values) == 0:
        return None
    return values.tail(N).mean()

def calc_home_advantage(team, current_date, df):
    """计算某队截至当前的主场胜率"""
    home_games = df[(df["home_team"] == team) & (df["date"] < current_date)]
    if len(home_games) == 0:
        return (df["home_score"] > df["away_score"]).mean()
    wins = (home_games["home_score"] > home_games["away_score"]).sum()
    return wins / len(home_games)

print("正在计算特征，这可能需要几分钟，请稍候...")

# 计算原有特征
df["home_form"] = df.apply(lambda row: calc_team_form(row["home_team"], row["date"], df), axis=1)
df["h2h_advantage"] = df.apply(lambda row: calc_h2h_advantage(row["home_team"], row["away_team"], row["date"], df), axis=1)

# 计算新增特征所需的联赛平均值（用于填充冷启动空值）
overall_home_goals_avg = df["home_score"].mean()
overall_away_concede_avg = df["away_score"].mean()

# 主队近5场平均进球
df["home_goals_avg"] = df.apply(
    lambda row: calc_rolling_avg(row["home_team"], row["date"], df, "home_score", is_home=True), axis=1
)
# 客队近5场平均失球
df["away_concede_avg"] = df.apply(
    lambda row: calc_rolling_avg(row["away_team"], row["date"], df, "away_score", is_home=False), axis=1
)

# 填充空值
df["home_goals_avg"] = df["home_goals_avg"].fillna(overall_home_goals_avg)
df["away_concede_avg"] = df["away_concede_avg"].fillna(overall_away_concede_avg)

# 攻防对比特征
df["attack_vs_defense"] = df["home_goals_avg"] - df["away_concede_avg"]

# 主队主场优势特征
df["home_advantage"] = df.apply(
    lambda row: calc_home_advantage(row["home_team"], row["date"], df), axis=1
)

# 目标变量：主队是否获胜
df["target"] = (df["home_score"] > df["away_score"]).astype(int)

# 保存特征数据
df.to_csv("featured_data.csv", index=False)
print("✅ 特征构造完成！已保存为 featured_data.csv")
print(f"数据维度：{df.shape}")
print(df[["home_team", "away_team", "home_form", "h2h_advantage", "attack_vs_defense", "home_advantage", "target"]].head(10))

# ========== 任务三：训练随机森林模型 ==========
features = ["home_form", "h2h_advantage", "attack_vs_defense", "home_advantage"]
X = df[features].fillna(0)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 40)
print(f"🎉 随机森林模型训练完成！")
print(f"📊 在测试集上的预测准确率为：{accuracy:.2%}")
print(f"🔍 特征重要性：")
importances = model.feature_importances_
for name, imp in zip(features, importances):
    print(f"   - {name}: {imp:.4f}")
print("=" * 40)

# 保存模型
joblib.dump(model, "football_predictor_model.pkl")
print("✅ 模型已保存为 football_predictor_model.pkl")
