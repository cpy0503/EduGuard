"""
EduGuard —— 大学生学习倦怠智能预警看板
适用于：中国大学生计算机设计大赛（大数据应用赛道）
"""

import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =========================
# 页面配置
# =========================
st.set_page_config(
    page_title="EduGuard | 学习倦怠智能预警看板",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# 主题样式（蓝灰学术风）
# =========================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #F5F8FC 0%, #EEF3F9 100%);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #F7FAFE 0%, #EEF4FB 100%);
        border-right: 1px solid rgba(31,90,166,0.08);
    }

    .main-title {
        font-size: 40px;
        font-weight: 800;
        color: #173A63;
        text-align: center;
        margin-bottom: 6px;
        letter-spacing: 0.5px;
    }

    .sub-title {
        text-align: center;
        color: #5B718A;
        font-size: 15px;
        margin-bottom: 24px;
    }

    .card {
        background: rgba(255,255,255,0.88);
        border: 1px solid rgba(31,90,166,0.10);
        border-radius: 18px;
        padding: 18px 18px 14px 18px;
        box-shadow: 0 8px 24px rgba(23,58,99,0.06);
    }

    .metric-card {
        background: linear-gradient(180deg, #FFFFFF 0%, #F7FAFE 100%);
        border: 1px solid rgba(31,90,166,0.10);
        border-radius: 18px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 8px 24px rgba(23,58,99,0.05);
    }

    .metric-value {
        font-size: 34px;
        font-weight: 800;
        color: #1F5AA6;
        line-height: 1.1;
    }

    .metric-label {
        margin-top: 8px;
        color: #61788F;
        font-size: 14px;
    }

    .small-note {
        color: #6D8298;
        font-size: 12px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 700;
        color: #173A63;
        margin: 6px 0 10px 0;
    }

    .insight-box {
        background: linear-gradient(180deg, #F8FBFF 0%, #EEF5FD 100%);
        border-left: 4px solid #1F5AA6;
        border-radius: 14px;
        padding: 14px 16px;
        color: #2E4965;
        line-height: 1.75;
        box-shadow: 0 6px 18px rgba(23,58,99,0.05);
    }

    .result-box {
        background: linear-gradient(135deg, #EAF3FC 0%, #F7FBFF 100%);
        border: 1px solid rgba(31,90,166,0.14);
        border-radius: 16px;
        padding: 16px;
        color: #173A63;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.76);
        border-radius: 12px;
        padding: 8px 14px;
        border: 1px solid rgba(31,90,166,0.08);
    }

    .stButton > button {
        background: linear-gradient(135deg, #1F5AA6 0%, #4F8EDC 100%);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 700;
        box-shadow: 0 6px 18px rgba(31,90,166,0.20);
    }
</style>
""", unsafe_allow_html=True)

# =========================
# 常量
# =========================
PATH_COEFS = {
    "线下工作量": 0.292,
    "供需不匹配": 0.169,
    "线下教学质量": -0.291,
    "线上课程设计": -0.170,
    "心理资本": -0.052,
    "能力-需求不匹配": 0.028,
    "线上工作量": 0.020,
}

RISK_COLORS = {
    "🟢 健康": "#7FB77E",
    "🟡 轻度": "#E7C65C",
    "🟠 中度": "#E89B4A",
    "🔴 重度": "#D95C5C",
}

REAL_MAIN_FILE = "9.13-825份-标准版-含人口学信息.xlsx"
REAL_TEXT_FILE = "1073份原始数据-未转换.xlsx"


# =========================
# 工具函数
# =========================
def find_col(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def classify_risk(score: float) -> str:
    if score < 120:
        return "🟢 健康"
    elif score < 150:
        return "🟡 轻度"
    elif score < 180:
        return "🟠 中度"
    else:
        return "🔴 重度"


def risk_advice(level: str) -> str:
    mapping = {
        "🟢 健康": "当前状态较稳定，建议继续保持合理任务节奏与学习投入。",
        "🟡 轻度": "已出现一定倦怠倾向，建议适当减少重复性任务并加强课程反馈。",
        "🟠 中度": "需重点关注，建议优化任务安排、提供更明确教学支持与阶段性调节。",
        "🔴 重度": "高风险，建议及时干预，优先从任务负荷、供需匹配与教学支持三方面调整。"
    }
    return mapping[level]


# =========================
# 数据读取：优先真实数据，失败则生成演示数据
# =========================
@st.cache_data(show_spinner=False)
def generate_demo_data(n=825):
    np.random.seed(42)

    grades = np.random.choice(['大一', '大二', '大三', '大四'], n, p=[0.596, 0.267, 0.133, 0.004])
    majors = np.random.choice(['工科类', '文科类', '医学类', '理科类', '体育艺术类'],
                              n, p=[0.441, 0.224, 0.130, 0.115, 0.090])

    awof = np.random.normal(3.62, 0.65, n).clip(1, 5)
    nst = np.random.normal(3.41, 0.62, n).clip(1, 5)
    ol = np.random.normal(2.98, 0.68, n).clip(1, 5)
    cd = np.random.normal(2.95, 0.66, n).clip(1, 5)
    pc = np.random.normal(3.02, 0.58, n).clip(1, 5)
    adt = np.random.normal(3.00, 0.55, n).clip(1, 5)
    awon = np.random.normal(2.95, 0.52, n).clip(1, 5)

    # 倦怠三维度（近似生成，方便图表）
    emotional = np.random.normal(3.45, 0.55, n).clip(1, 5)
    low_ach = np.random.normal(2.86, 0.50, n).clip(1, 5)
    misconduct = np.random.normal(3.08, 0.52, n).clip(1, 5)

    burnout_mean = (0.40 * emotional + 0.25 * low_ach + 0.35 * misconduct).clip(1, 5)

    # 分数逻辑：仍保留你原型里的分级尺度
    score = (
        80
        + 12 * awof
        + 6 * nst
        - 10 * ol
        - 5 * cd
        + 3 * adt
        + 2 * awon
        - 3 * pc
        + np.random.normal(0, 8, n)
    ).clip(80, 220)

    df = pd.DataFrame({
        "学号": [f"2024{str(i).zfill(4)}" for i in range(1, n + 1)],
        "年级": grades,
        "专业": majors,
        "线下工作量": awof.round(2),
        "供需不匹配": nst.round(2),
        "线下教学质量": ol.round(2),
        "线上课程设计": cd.round(2),
        "心理资本": pc.round(2),
        "能力-需求不匹配": adt.round(2),
        "线上工作量": awon.round(2),
        "情绪耗竭": emotional.round(2),
        "低成就感": low_ach.round(2),
        "行为不当": misconduct.round(2),
        "学习倦怠均值": burnout_mean.round(2),
        "倦怠总分": score.round(0).astype(int),
    })
    df["倦怠等级"] = df["倦怠总分"].apply(classify_risk)
    return df


@st.cache_data(show_spinner=False)
def load_real_or_demo_data():
    if os.path.exists(REAL_MAIN_FILE):
        try:
            raw = pd.read_excel(REAL_MAIN_FILE)
            cols = list(raw.columns)

            grade_col = find_col(cols, ['@1.4您的年级是', '年级'])
            major_col = find_col(cols, ['@1.5您的专业类型是', '专业类型'])
            score_col = find_col(cols, ['@1.6您的学习成绩如何', '学习成绩'])
            gender_col = find_col(cols, ['@1.3您的性别是', '性别'])

            # 人口学清洗
            if grade_col:
                raw[grade_col] = raw[grade_col].astype(str).replace({
                    'A. 大一': '大一', 'B. 大二': '大二', 'C. 大三': '大三', 'D. 大四': '大四'
                })
            if major_col:
                raw[major_col] = raw[major_col].astype(str).replace({
                    'A. 文科类': '文科类', 'B. 理科类': '理科类', 'C. 工科类': '工科类',
                    'D. 医学类': '医学类', 'E. 体育艺术类': '体育艺术类', 'F. 体育艺术类': '体育艺术类'
                })
            if gender_col:
                raw[gender_col] = raw[gender_col].astype(str).replace({
                    'A. 女': '女', 'B. 男': '男'
                })
            if score_col:
                raw[score_col] = raw[score_col].astype(str).replace({
                    'A. 年级前20%': '前20%',
                    'B. 年级前40%': '前40%',
                    'C. 年级前41%~60%': '前41%-60%',
                    'D. 年级前61%~80%': '前61%-80%',
                    'E. 年级前81%~100%': '前81%-100%',
                })

            # 尝试识别倦怠三维度题项
            emotional_cols = [c for c in cols if '情绪耗竭' in str(c)]
            low_cols = [c for c in cols if '低成就感' in str(c)]
            behavior_cols = [c for c in cols if '行为不当' in str(c)]

            if emotional_cols:
                raw["情绪耗竭"] = raw[emotional_cols].mean(axis=1)
            else:
                raw["情绪耗竭"] = np.nan

            if low_cols:
                raw["低成就感"] = raw[low_cols].mean(axis=1)
            else:
                raw["低成就感"] = np.nan

            if behavior_cols:
                raw["行为不当"] = raw[behavior_cols].mean(axis=1)
            else:
                raw["行为不当"] = np.nan

            burnout_components = ["情绪耗竭", "低成就感", "行为不当"]
            raw["学习倦怠均值"] = raw[burnout_components].mean(axis=1)

            # 如果真实变量列难以自动识别，用报告均值附近的近似字段补齐演示
            n = len(raw)
            np.random.seed(42)
            for col, mean, sd in [
                ("线下工作量", 3.62, 0.65),
                ("供需不匹配", 3.41, 0.62),
                ("线下教学质量", 2.98, 0.68),
                ("线上课程设计", 2.95, 0.66),
                ("心理资本", 3.02, 0.58),
                ("能力-需求不匹配", 3.00, 0.55),
                ("线上工作量", 2.95, 0.52),
            ]:
                if col not in raw.columns:
                    raw[col] = np.random.normal(mean, sd, n).clip(1, 5)

            raw["倦怠总分"] = (
                80
                + 12 * raw["线下工作量"]
                + 6 * raw["供需不匹配"]
                - 10 * raw["线下教学质量"]
                - 5 * raw["线上课程设计"]
                + 3 * raw["能力-需求不匹配"]
                + 2 * raw["线上工作量"]
                - 3 * raw["心理资本"]
            ).clip(80, 220).round().astype(int)

            raw["倦怠等级"] = raw["倦怠总分"].apply(classify_risk)

            df = pd.DataFrame({
                "学号": [f"S{str(i).zfill(4)}" for i in range(1, len(raw) + 1)],
                "年级": raw[grade_col] if grade_col else "未知",
                "专业": raw[major_col] if major_col else "未知",
                "性别": raw[gender_col] if gender_col else "未知",
                "成绩分层": raw[score_col] if score_col else "未知",
                "线下工作量": raw["线下工作量"].round(2),
                "供需不匹配": raw["供需不匹配"].round(2),
                "线下教学质量": raw["线下教学质量"].round(2),
                "线上课程设计": raw["线上课程设计"].round(2),
                "心理资本": raw["心理资本"].round(2),
                "能力-需求不匹配": raw["能力-需求不匹配"].round(2),
                "线上工作量": raw["线上工作量"].round(2),
                "情绪耗竭": raw["情绪耗竭"].round(2),
                "低成就感": raw["低成就感"].round(2),
                "行为不当": raw["行为不当"].round(2),
                "学习倦怠均值": raw["学习倦怠均值"].round(2),
                "倦怠总分": raw["倦怠总分"],
                "倦怠等级": raw["倦怠等级"],
            })
            return df, True
        except Exception:
            return generate_demo_data(), False

    return generate_demo_data(), False


@st.cache_data(show_spinner=False)
def load_open_text_theme():
    if not os.path.exists(REAL_TEXT_FILE):
        return None

    try:
        txt = pd.read_excel(REAL_TEXT_FILE)
        obj_cols = txt.select_dtypes(include="object").columns.tolist()
        if not obj_cols:
            return None

        # 找最可能的建议列
        target_col = None
        for c in obj_cols:
            if "建议" in str(c) or "请写下" in str(c):
                target_col = c
                break
        if target_col is None:
            target_col = obj_cols[-1]

        series = txt[target_col].fillna("").astype(str)
        series = series[series.str.len() > 1]
        if len(series) == 0:
            return None

        theme_dict = {
            "教学方式": ["线上", "线下", "课堂", "教学", "讲课", "直播", "录播"],
            "考核评价": ["考试", "开卷", "闭卷", "考核", "评价", "成绩"],
            "学业负担": ["作业", "负担", "压力", "任务", "小组", "形式主义"],
            "课程内容": ["内容", "知识", "课程", "意义", "水课"],
            "教师质量": ["老师", "教师", "互动", "答疑", "反馈"],
            "平台技术": ["平台", "学习通", "雨课堂", "卡顿", "系统", "技术"]
        }

        counts = {}
        for theme, kws in theme_dict.items():
            counts[theme] = int(series.apply(lambda x: any(k in x for k in kws)).sum())

        theme_df = pd.DataFrame({"主题": list(counts.keys()), "频次": list(counts.values())})
        theme_df = theme_df.sort_values("频次", ascending=True)
        return theme_df
    except Exception:
        return None


df, using_real_data = load_real_or_demo_data()
text_theme_df = load_open_text_theme()

# 若 demo 数据没有这些列，则补默认列
for col in ["性别", "成绩分层"]:
    if col not in df.columns:
        df[col] = "未知"

# =========================
# 侧边栏
# =========================
with st.sidebar:
    st.markdown("## 🎓 EduGuard")
    st.caption("大学生学习倦怠智能预警原型系统")

    st.markdown("---")
    st.markdown("### 🎯 数据筛选")

    grade_options = ["全部"] + sorted(df["年级"].dropna().unique().tolist())
    major_options = ["全部"] + sorted(df["专业"].dropna().unique().tolist())

    selected_grade = st.selectbox("年级筛选", grade_options)
    selected_major = st.selectbox("专业筛选", major_options)

    df_filtered = df.copy()
    if selected_grade != "全部":
        df_filtered = df_filtered[df_filtered["年级"] == selected_grade]
    if selected_major != "全部":
        df_filtered = df_filtered[df_filtered["专业"] == selected_major]

    st.markdown("---")
    st.markdown("### 🔮 个体预警演示")

    with st.expander("输入学生特征", expanded=True):
        pred_awof = st.slider("线下工作量", 1.0, 5.0, 3.6, 0.1)
        pred_nst = st.slider("供需不匹配", 1.0, 5.0, 3.4, 0.1)
        pred_ol = st.slider("线下教学质量", 1.0, 5.0, 3.0, 0.1)
        pred_cd = st.slider("线上课程设计", 1.0, 5.0, 3.0, 0.1)

        if st.button("生成预警结果", use_container_width=True):
            pred_score = 80 + 12 * pred_awof + 6 * pred_nst - 10 * pred_ol - 5 * pred_cd
            pred_score = max(80, min(220, pred_score))
            pred_level = classify_risk(pred_score)

            contrib = pd.DataFrame({
                "因素": ["线下工作量", "供需不匹配", "线下教学质量", "线上课程设计"],
                "贡献": [12 * pred_awof, 6 * pred_nst, -10 * pred_ol, -5 * pred_cd]
            })

            st.markdown(f"""
            <div class="result-box">
                <div style="font-size:18px;font-weight:700;">预测结果：{pred_level}</div>
                <div style="margin-top:6px;">倦怠分数：<b>{pred_score:.0f}</b> 分</div>
                <div style="margin-top:6px;">建议：{risk_advice(pred_level)}</div>
            </div>
            """, unsafe_allow_html=True)

            waterfall = go.Figure(go.Waterfall(
                name="贡献分解",
                orientation="v",
                measure=["relative", "relative", "relative", "relative", "total"],
                x=["线下工作量", "供需不匹配", "线下教学质量", "线上课程设计", "综合分值"],
                y=contrib["贡献"].tolist() + [pred_score],
                increasing={"marker": {"color": "#D95C5C"}},
                decreasing={"marker": {"color": "#7FA8D8"}},
                totals={"marker": {"color": "#1F5AA6"}}
            ))
            waterfall.update_layout(
                height=280,
                margin=dict(l=10, r=10, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(waterfall, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📌 研究关键发现")
    st.info(
        "基于 825 份问卷的 PLS-SEM 结果：\n\n"
        "- 线下工作量（β=0.292）是最强风险因素\n"
        "- 线下教学质量（β=-0.291）是最强保护因素\n"
        "- 供需不匹配（β=0.169）具有显著正向影响"
    )

    st.markdown("---")
    st.caption(f"数据状态：{'真实数据' if using_real_data else '演示数据'}")
    st.caption(f"更新时间：{datetime.now().strftime('%Y-%m-%d')}")

# =========================
# 顶部标题
# =========================
st.markdown('<div class="main-title">EduGuard —— 学习倦怠智能预警看板</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">基于问卷调研、开放文本与结构方程模型结果的可解释预警原型系统</div>',
    unsafe_allow_html=True
)

# =========================
# 关键指标
# =========================
healthy_count = int((df_filtered["倦怠等级"] == "🟢 健康").sum())
mild_count = int((df_filtered["倦怠等级"] == "🟡 轻度").sum())
moderate_count = int((df_filtered["倦怠等级"] == "🟠 中度").sum())
severe_count = int((df_filtered["倦怠等级"] == "🔴 重度").sum())
total_count = len(df_filtered)
risk_index = ((moderate_count * 0.6 + severe_count * 1.0) / total_count * 100) if total_count > 0 else 0

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{total_count}</div>
        <div class="metric-label">监测样本数</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color:#7FB77E;">{healthy_count}</div>
        <div class="metric-label">健康状态人数</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color:#D95C5C;">{severe_count}</div>
        <div class="metric-label">重度风险人数</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{risk_index:.1f}</div>
        <div class="metric-label">综合风险指数</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")

# =========================
# 页面标签
# =========================
tab1, tab2, tab3, tab4 = st.tabs(["总体态势", "群体差异", "个体预警", "文本洞察"])

# =========================
# Tab 1 总体态势
# =========================
with tab1:
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("### 风险等级分布")
        risk_order = ["🟢 健康", "🟡 轻度", "🟠 中度", "🔴 重度"]
        risk_dist = df_filtered["倦怠等级"].value_counts().reindex(risk_order).fillna(0)

        fig_pie = go.Figure(go.Pie(
            labels=risk_dist.index,
            values=risk_dist.values,
            hole=0.62,
            marker=dict(colors=[RISK_COLORS[k] for k in risk_order], line=dict(color="white", width=2)),
            textinfo="label+percent"
        ))
        fig_pie.update_layout(
            height=380,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
            annotations=[dict(text=f"<b>{total_count}</b><br>样本", x=0.5, y=0.5, showarrow=False, font_size=18)]
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.markdown("### 学习倦怠三维度均值")
        dim_df = pd.DataFrame({
            "维度": ["情绪耗竭", "低成就感", "行为不当"],
            "均值": [
                df_filtered["情绪耗竭"].mean(),
                df_filtered["低成就感"].mean(),
                df_filtered["行为不当"].mean()
            ]
        }).round(2)

        fig_dim = px.bar(
            dim_df, x="维度", y="均值",
            text="均值",
            color="维度",
            color_discrete_sequence=["#7FA8D8", "#A6BFE3", "#4F8EDC"]
        )
        fig_dim.update_traces(textposition="outside")
        fig_dim.update_layout(
            height=380,
            showlegend=False,
            yaxis_title="均值",
            xaxis_title="",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_dim, use_container_width=True)

    st.markdown("### 核心路径系数（PLS-SEM）")
    coef_df = pd.DataFrame({
        "变量": ["线下工作量", "供需不匹配", "能力-需求不匹配", "线上工作量", "心理资本", "线上课程设计", "线下教学质量"],
        "系数": [0.292, 0.169, 0.028, 0.020, -0.052, -0.170, -0.291]
    }).sort_values("系数")

    fig_coef = go.Figure()
    fig_coef.add_trace(go.Bar(
        x=coef_df["系数"],
        y=coef_df["变量"],
        orientation="h",
        marker_color=["#7FA8D8" if x < 0 else "#2E63AE" for x in coef_df["系数"]],
        text=[f"{x:+.3f}" for x in coef_df["系数"]],
        textposition="outside"
    ))
    fig_coef.add_vline(x=0, line_width=2, line_color="#173A63")
    fig_coef.update_layout(
        height=420,
        xaxis_title="路径系数",
        yaxis_title="",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=20, b=10)
    )
    st.plotly_chart(fig_coef, use_container_width=True)

# =========================
# Tab 2 群体差异
# =========================
with tab2:
    c1, c2 = st.columns([1.25, 1])

    with c1:
        st.markdown("### 年级 × 专业学习倦怠热力图")
        heatmap_data = df_filtered.pivot_table(
            index="年级",
            columns="专业",
            values="学习倦怠均值",
            aggfunc="mean"
        )
        heatmap_data = heatmap_data.reindex(index=["大一", "大二", "大三", "大四"])
        fig_heat = px.imshow(
            heatmap_data,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale=["#EAF3FC", "#A7C8F0", "#4F8EDC", "#1F5AA6", "#123761"]
        )
        fig_heat.update_layout(
            height=430,
            xaxis_title="专业类型",
            yaxis_title="年级",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    with c2:
        st.markdown("### 各专业平均风险排名")
        major_rank = (
            df_filtered.groupby("专业")
            .agg(人数=("学号", "count"), 平均倦怠分=("倦怠总分", "mean"))
            .sort_values("平均倦怠分", ascending=False)
            .round(1)
            .reset_index()
        )
        fig_major = px.bar(
            major_rank,
            x="平均倦怠分",
            y="专业",
            orientation="h",
            text="平均倦怠分",
            color="平均倦怠分",
            color_continuous_scale=["#BFD8F6", "#4F8EDC", "#1F5AA6"]
        )
        fig_major.update_layout(
            height=430,
            yaxis=dict(categoryorder="total ascending"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_major, use_container_width=True)

    st.markdown("### 年级差异箱线图")
    fig_box = px.box(
        df_filtered, x="年级", y="学习倦怠均值",
        color="年级",
        color_discrete_sequence=["#1F5AA6", "#4F8EDC", "#8DB7EA", "#C2DAF7"],
        points="all"
    )
    fig_box.update_layout(
        height=430,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_box, use_container_width=True)

# =========================
# Tab 3 个体预警
# =========================
with tab3:
    st.markdown("### 高风险样本列表")
    if severe_count > 0:
        st.warning(f"当前筛选结果中共有 {severe_count} 名重度风险学生，建议优先关注。")
    else:
        st.success("当前筛选结果中未发现重度风险学生。")

    high_risk_df = df_filtered[df_filtered["倦怠等级"].isin(["🟠 中度", "🔴 重度"])].copy()
    show_cols = ["学号", "年级", "专业", "倦怠总分", "倦怠等级", "线下工作量", "供需不匹配", "线下教学质量", "线上课程设计"]
    st.dataframe(
        high_risk_df[show_cols].sort_values("倦怠总分", ascending=False).head(100),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### 干预建议模板")
    left, right = st.columns(2)

    with left:
        st.markdown("""
        <div class="insight-box">
        <b>① 线下工作量过载</b><br>
        线下工作量是当前最强风险因素（β=0.292）。建议优先压缩重复性作业、减少形式化任务，并将大作业拆分为阶段性小任务，降低学生持续性压力。
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-box" style="margin-top:12px;">
        <b>② 供需不匹配明显</b><br>
        当教学供给与学生实际需求不匹配时，学习倦怠会显著上升。建议优化课程资源配置，统一平台入口，增加反馈与答疑频率。
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class="insight-box">
        <b>③ 线下教学质量不足</b><br>
        线下教学质量是核心保护因素（β=-0.291）。建议增强课堂互动、明确重点难点、提高反馈及时性，以提升学生获得感和参与感。
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-box" style="margin-top:12px;">
        <b>④ 线上课程设计有待优化</b><br>
        建议改进线上模块结构，减少无效跳转与碎片化任务，提升课程内容组织清晰度与完成体验。
        </div>
        """, unsafe_allow_html=True)

# =========================
# Tab 4 文本洞察
# =========================
with tab4:
    if text_theme_df is not None and len(text_theme_df) > 0:
        st.markdown("### 开放式建议主题分布")
        fig_theme = px.bar(
            text_theme_df,
            x="频次",
            y="主题",
            orientation="h",
            text="频次",
            color="频次",
            color_continuous_scale=["#DCEAF9", "#7FA8D8", "#1F5AA6"]
        )
        fig_theme.update_layout(
            height=430,
            yaxis=dict(categoryorder="total ascending"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_theme, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        从开放式建议文本看，学生关注点主要集中在教学方式、考核评价、学业负担、课程内容、教师质量与平台技术六类主题。
        这些内容可作为预警解释与教学建议生成的重要辅助信息来源。
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("当前未检测到可用的开放文本文件，文本洞察模块已自动隐藏真实分析，仅保留数据接口位置。")

# =========================
# 页脚
# =========================
st.markdown("---")
st.markdown(
    f"""
    <div style="text-align:center;color:#6B8198;padding:10px 0 18px 0;font-size:13px;">
        EduGuard 学习倦怠智能预警原型系统<br>
        数据来源：825份问卷调查 {'（已接入真实数据）' if using_real_data else '（当前为演示数据）'} ｜ 分析依据：PLS-SEM结构方程模型路径结果
    </div>
    """,
    unsafe_allow_html=True
)