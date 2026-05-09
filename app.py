# 在 app.py 顶部导入新模块
from data_fetcher import fetch_live_fixtures


import streamlit as st
import pandas as pd
import joblib

# ---------- 页面配置 ----------
st.set_page_config(page_title="足球比赛结果预测", page_icon="⚽", layout="centered")

st.title("⚽ 英超比赛结果预测")
st.markdown("基于历史数据的随机森林模型，预测主队胜、平、负的概率。")

# ---------- 加载模型和数据 ----------
@st.cache_resource
def load_model():
    return joblib.load("football_predictor_model.pkl")


@st.cache_data(ttl=600) # 缓存10分钟，减少API调用次数
def load_data():
    # 1. 先加载本地历史数据
    df = pd.read_csv("featured_data.csv")
    
    # 2. 获取最新赛季数据 (比如2025)
    df_live = fetch_live_fixtures(season="2025")
    
    # 3. 合并数据并去重
    if df_live is not None and not df_live.empty:
        df = pd.concat([df, df_live], ignore_index=True)
        df = df.drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")
        df = df.sort_values("date")
        print("✅ 实时数据加载成功。")
    else:
        print("⚠️ 警告：无法获取最新数据，将使用现有历史数据。")
    
    teams = sorted(set(df["home_team"].unique()) | set(df["away_team"].unique()))
    return df, teams
model = load_model()
df,team_list = load_data()
# ---------- 特征计算函数（与 build_model.py 保持一致）----------
def calc_team_form(team, current_date, df, N=5):
    home_games = df[(df["home_team"] == team) & (df["date"] < current_date)].copy()
    away_games = df[(df["away_team"] == team) & (df["date"] < current_date)].copy()
    home_games["win"] = home_games["home_score"] > home_games["away_score"]
    away_games["win"] = away_games["away_score"] > away_games["home_score"]
    all_games = pd.concat([home_games, away_games]).sort_values("date", ascending=False)
    last_N = all_games.head(N)
    if len(last_N) == 0:
        return (df["home_score"] > df["away_score"]).mean()
    return last_N["win"].sum() / len(last_N)

def calc_h2h_advantage(home, away, current_date, df, N=3):
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
    home_games = df[(df["home_team"] == team) & (df["date"] < current_date)]
    if len(home_games) == 0:
        return (df["home_score"] > df["away_score"]).mean()
    wins = (home_games["home_score"] > home_games["away_score"]).sum()
    return wins / len(home_games)

# ---------- 构造单场比赛的特征 ----------
def build_match_features(home, away, df):
    # 使用最新日期作为“当前日期”，模拟实时预测
    current_date = df["date"].max()
    
    home_form = calc_team_form(home, current_date, df)
    h2h_adv = calc_h2h_advantage(home, away, current_date, df)
    
    # 攻防特征
    home_goals = calc_rolling_avg(home, current_date, df, "home_score", is_home=True)
    away_concede = calc_rolling_avg(away, current_date, df, "away_score", is_home=False)
    overall_home_goals = df["home_score"].mean()
    overall_away_concede = df["away_score"].mean()
    if home_goals is None:
        home_goals = overall_home_goals
    if away_concede is None:
        away_concede = overall_away_concede
    attack_vs_defense = home_goals - away_concede
    
    home_adv = calc_home_advantage(home, current_date, df)
    
    return pd.DataFrame([[
        home_form, h2h_adv, attack_vs_defense, home_adv
    ]], columns=["home_form", "h2h_advantage", "attack_vs_defense", "home_advantage"])

# ---------- 界面交互 ----------
col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("🏠 主队", team_list, index=team_list.index("Arsenal FC") if "Arsenal FC" in team_list else 0)
with col2:
    away_team = st.selectbox("🛫 客队", team_list, index=team_list.index("Manchester City FC") if "Manchester City FC" in team_list else 1)

if st.button("🔮 开始预测", type="primary"):
    with st.spinner("正在计算特征并预测..."):
        X_new = build_match_features(home_team, away_team, df)
        
        # 预测概率（随机森林输出各类别概率）
        proba = model.predict_proba(X_new)[0]
        # 类别顺序：0 = 主队不胜（平或负），1 = 主队胜
        # 但我们需要胜、平、负三个概率，这里用简化处理：用主胜概率作为胜，剩余概率按主队历史平负比例分配
        # 更准确做法：训练时用三分类，这里为了演示，用二分类概率拆分展示
        win_prob = proba[1]
        # 粗略估计平、负概率（基于联赛历史分布）
        total_matches = len(df)
        draw_rate = len(df[df["home_score"] == df["away_score"]]) / total_matches
        lose_rate = 1 - (df["home_score"] > df["away_score"]).mean() - draw_rate
        
        not_win_prob = 1 - win_prob
        draw_prob = not_win_prob * (draw_rate / (draw_rate + lose_rate))
        lose_prob = not_win_prob * (lose_rate / (draw_rate + lose_rate))
        
        st.success("预测完成！")
        st.metric("主队获胜概率", f"{win_prob:.1%}")
        col_a, col_b = st.columns(2)
        col_a.metric("平局概率", f"{draw_prob:.1%}")
        col_b.metric("客队获胜概率", f"{lose_prob:.1%}")
        
        # 显示特征值（可选，用于演示）
        with st.expander("查看特征值"):
            X_display = X_new.rename(columns = {'home_form':'主队近期胜率','h2h_advantage':'历史交战优势','attack_vs_defense':'进球效率','home_advantage':'主场优势'})
            st.dataframe(X_display)

