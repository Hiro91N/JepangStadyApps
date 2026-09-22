import streamlit as st
import openai
import csv
import io
import json
import random
from datetime import datetime, timedelta
from collections import defaultdict

# ============================================================
# 設定
# ============================================================
st.set_page_config(
    page_title="日本語AI学習システム",
    page_icon="🎌",
    layout="wide"
)

# OpenAI APIキー（Secretsから取得）
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ============================================================
# カテゴリ定義
# ============================================================
GRAMMAR_CATEGORIES = {
    "particles_wa_ga": "助詞「は・が」",
    "particles_ni_de": "助詞「に・で・を」",
    "particles_to_ya": "助詞「と・や・の」",
    "verb_group1": "動詞の活用（グループ1）",
    "verb_group2": "動詞の活用（グループ2）",
    "verb_group3": "動詞の活用（グループ3）",
    "te_form": "〜て形",
    "ta_form": "〜た形",
    "nai_form": "〜ない形",
    "adjectives_i": "い形容詞",
    "adjectives_na": "な形容詞",
    "past_tense": "過去形",
    "questions": "疑問文",
    "counting": "数え方・助数詞",
    "existence": "ある・いる",
    "want_to": "〜たい（希望）",
    "can_do": "〜ことができる（可能）",
    "desu_masu": "です・ます体",
}

VOCAB_CATEGORIES = {
    "food_drink": "食べ物・飲み物",
    "daily_actions": "日常の動作",
    "places": "場所",
    "time_date": "時間・日付",
    "family_people": "家族・人間関係",
    "adjectives": "形容詞・イメージ",
    "body_parts": "体の部位",
    "nature": "自然・天気",
    "transportation": "交通・移動",
    "shopping": "買い物・お金",
    "colors": "色",
    "emotions": "感情・気持ち",
    "work_study": "仕事・学校",
    "home_items": "家・家具",
}

ALL_CATEGORIES = {**GRAMMAR_CATEGORIES, **VOCAB_CATEGORIES}

# ============================================================
# SM-2 エビングハウス忘却曲線
# ============================================================
def sm2_calculate(quality, reps=0, ef=2.5, interval=0):
    """
    SM-2アルゴリズム
    quality: 0-5 (5=完璧, 0=全くわからない)
    reps: 連続正解回数
    ef: 容易度係数
    interval: 前回の間隔（日）
    """
    if quality < 3:
        reps = 0
        interval = 1
    else:
        if reps == 0:
            interval = 1
        elif reps == 1:
            interval = 6
        else:
            interval = round(interval * ef)
        reps += 1

    ef = ef + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    ef = max(1.3, ef)

    return reps, ef, interval


def get_weak_categories(history, top_n=5):
    """苦手なカテゴリを抽出（正解率が低い順）"""
    if not history:
        return []

    cat_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for row in history:
        cat = row.get("category", "")
        if cat:
            cat_stats[cat]["total"] += 1
            if row.get("is_correct"):
                cat_stats[cat]["correct"] += 1

    results = []
    for cat, stats in cat_stats.items():
        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        if stats["total"] >= 2:
            results.append((cat, accuracy, stats["total"]))

    results.sort(key=lambda x: x[1])
    return [r[0] for r in results[:top_n]]


# ============================================================
# OpenAI API ヘルパー
# ============================================================
def call_openai(prompt, temperature=0.7, max_tokens=4000):
    """GPT-4o mini を呼び出す"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "あなたは日本語教師です。インドネシ語と日本語の両方で説明できます。HTML形式で出力してください。"},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"APIエラー: {e}")
        return None


# ============================================================
# プリント生成（文法＋単語統合）
# ============================================================
def generate_worksheet(level, theme, num_questions, history=None):
    """学習プリントを生成（文法＋単語統合）"""

    weak_cats = get_weak_categories(history) if history else []

    prompt = f"""日本語学習者向けの学習プリントを作成してください。

【条件】
- レベル: {level}
- テーマ: {theme}
- 問題数: {num_questions}問
- 形式: 文法問題と単語問題を統合して連続番号（1〜{num_questions}）で出題
- 文法問題: 助詞選択・動詞活用・形容詞活用など
- 単語問題: 日本語→インドネシア語またはインドネシア語→日本語の単語テスト
- 解説言語: インドネシア語（Bahasa Indonesia）

【重点出題カテゴリ】
"""
    if weak_cats:
        prompt += "以下の苦手な分野を70%の比率で重点的に出題してください：\n"
        for cat in weak_cats[:5]:
            prompt += f"- {ALL_CATEGORIES.get(cat, cat)}\n"
    else:
        prompt += "初回学習のため、初級文法と基礎単語を均等に出題してください。\n"

    prompt += """
【出力形式】
以下のHTMLを生成してください（CSSは含めないでください、style属性も使わないでください）：

```html
<h1>日本語学習プリント</h1>
<p>レベル: XXX | テーマ: XXX | 日付: YYYY-MM-DD</p>

<h2>問題</h2>
<ol>
  <li>【文法】問題文<br>答え: ________</li>
  <li>【単語】問題文<br>答え: ________</li>
  ...
</ol>

<h2>解答・解説</h2>
<ol>
  <li>答え: XXX<br>解説: （インドネシア語で詳細な解説）</li>
  ...
</ol>
```

必ず問題と解答・解説の両方を含めてください。
"""

    return call_openai(prompt, temperature=0.8)


# ============================================================
# 復習テストプリント生成
# ============================================================
def generate_review_worksheet(history, num_questions=10):
    """苦手分野重点の復習テストプリントを生成"""

    weak_cats = get_weak_categories(history, top_n=5)

    if not weak_cats:
        st.warning("まだ学習履歴がありません。先に理解度テストを受けてください。")
        return None

    # 苦手カテゴリの詳細情報を集める
    cat_details = []
    for cat in weak_cats:
        cat_rows = [r for r in history if r.get("category") == cat]
        total = len(cat_rows)
        correct = sum(1 for r in cat_rows if r.get("is_correct"))
        accuracy = (correct / total * 100) if total > 0 else 0
        cat_details.append({
            "key": cat,
            "name": ALL_CATEGORIES.get(cat, cat),
            "total": total,
            "correct": correct,
            "accuracy": accuracy
        })

    prompt = f"""日本語学習者向けの「復習テストプリント」を作成してください。

【学習者の苦手な分野】
"""
    for d in cat_details:
        prompt += f"- {d['name']}: 正解率 {d['accuracy']:.0f}%（{d['correct']}/{d['total']}問）\n"

    prompt += f"""
【条件】
- 問題数: {num_questions}問（8〜12問）
- 上記の苦手な分野のみを出題
- 問題形式: 文法問題と単語問題を統合
- 解説言語: インドネシア語（Bahasa Indonesia）
- 各問題の解説に「なぜ間違えやすいか」「記憶のコツ」を含める

【出力形式】
以下のHTMLを生成してください（CSSは含めないでください）：

```html
<h1>復習テストプリント</h1>
<p>重点分野: XXX, XXX, XXX</p>

<h2>問題</h2>
<ol>
  <li>【XXX】問題文<br>答え: ________</li>
  ...
</ol>

<h2>解答・解説</h2>
<ol>
  <li>答え: XXX<br>解説: （インドネシア語で詳細な解説・記憶のコツ）</li>
  ...
</ol>
```
"""

    return call_openai(prompt, temperature=0.8)


def generate_review_advice(history):
    """復習プリント用のAI評価アドバイス（バイリンガル）"""

    weak_cats = get_weak_categories(history, top_n=5)
    cat_details = []
    for cat in weak_cats:
        cat_rows = [r for r in history if r.get("category") == cat]
        total = len(cat_rows)
        correct = sum(1 for r in cat_rows if r.get("is_correct"))
        accuracy = (correct / total * 100) if total > 0 else 0
        cat_details.append({
            "name": ALL_CATEGORIES.get(cat, cat),
            "accuracy": accuracy,
            "total": total
        })

    prompt = f"""以下の学習データに基づいて、学習者への評価アドバイスを作成してください。

【学習データ】
- 総回答数: {len(history)}問
- 苦手な分野:
"""
    for d in cat_details:
        prompt += f"  - {d['name']}: 正解率 {d['accuracy']:.0f}%（{d['total']}問）\n"

    prompt += """
【出力条件】
以下の5セクションについて、それぞれ「インドネシア語」と「日本語」で同じ内容を出力してください。
形式は以下の通り：

```html
<div class="bilingual-section">
  <div class="id">
    <h3>🇮🇩 [セクション名（インドネシア語）]</h3>
    <p>[内容（インドネシア語）]</p>
  </div>
  <div class="ja">
    <h3>🇯🇵 [セクション名（日本語）]</h3>
    <p>[内容（日本語）]</p>
  </div>
</div>
```

【5セクション】
1. 総合評価（Penilaian Keseluruhan / 総合評価）
2. 強み（Kelebihan / 強み）
3. 改善点（Area yang Perlu Ditingkatkan / 改善点）
4. 次のステップ（Langkah Selanjutnya / 次のステップ）
5. 学習モチベーション（Semangat Belajar / 学習モチベーション）

内容は具体的に、データに基づいて作成してください。励ましの言葉を含めてください。
"""

    return call_openai(prompt, temperature=0.7, max_tokens=3000)


# ============================================================
# 理解度テスト
# ============================================================
def generate_test_questions(categories, num_questions=5):
    """理解度テストの問題を生成"""

    cat_list = ", ".join([ALL_CATEGORIES.get(c, c) for c in categories])

    prompt = f"""日本語学習者向けの理解度テストを作成してください。

【出題分野】{cat_list}
【問題数】{num_questions}問
【形式】4択問題（A/B/C/D）
【解説言語】インドネシア語

各問題に以下を含めてください：
- 問題文
- 4つの選択肢（A/B/C/D）
- 正解の選択肢
- インドネシア語での解説

以下のJSON形式で出力してください：

```json
[
  {{
    "category": "カテゴリキー",
    "question": "問題文",
    "choices": {{"A": "選択肢1", "B": "選択肢2", "C": "選択肢3", "D": "選択肢4"}},
    "correct": "A",
    "explanation": "解説（インドネシア語）"
  }}
]
```
"""

    response = call_openai(prompt, temperature=0.8)
    if not response:
        return []

    try:
        # JSON部分を抽出
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0].strip()
        else:
            json_str = response.strip()

        questions = json.loads(json_str)
        return questions
    except Exception as e:
        st.error(f"問題のパースに失敗しました: {e}")
        return []


# ============================================================
# 評価シート生成
# ============================================================
def generate_evaluation_sheet(history):
    """評価シートを生成（バイリンガルAI評価付き）"""

    if not history:
        st.warning("まだ学習履歴がありません。先に理解度テストを受けてください。")
        return None

    # カテゴリ別統計
    cat_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for row in history:
        cat = row.get("category", "")
        if cat:
            cat_stats[cat]["total"] += 1
            if row.get("is_correct"):
                cat_stats[cat]["correct"] += 1

    # 文法・単語別統計
    grammar_correct = sum(1 for r in history if r.get("category") in GRAMMAR_CATEGORIES and r.get("is_correct"))
    grammar_total = sum(1 for r in history if r.get("category") in GRAMMAR_CATEGORIES)
    vocab_correct = sum(1 for r in history if r.get("category") in VOCAB_CATEGORIES and r.get("is_correct"))
    vocab_total = sum(1 for r in history if r.get("category") in VOCAB_CATEGORIES)

    grammar_acc = (grammar_correct / grammar_total * 100) if grammar_total > 0 else 0
    vocab_acc = (vocab_correct / vocab_total * 100) if vocab_total > 0 else 0

    # 復習スケジュール
    today = datetime.now().date()
    review_schedule = []
    for cat, stats in cat_stats.items():
        if stats["total"] >= 2:
            accuracy = stats["correct"] / stats["total"]
            if accuracy < 0.7:
                review_schedule.append({
                    "category": ALL_CATEGORIES.get(cat, cat),
                    "accuracy": accuracy * 100,
                    "next_review": "明日",
                    "priority": "高"
                })
            elif accuracy < 0.85:
                review_schedule.append({
                    "category": ALL_CATEGORIES.get(cat, cat),
                    "accuracy": accuracy * 100,
                    "next_review": "3日後",
                    "priority": "中"
                })
            else:
                review_schedule.append({
                    "category": ALL_CATEGORIES.get(cat, cat),
                    "accuracy": accuracy * 100,
                    "next_review": "1週間後",
                    "priority": "低"
                })

    review_schedule.sort(key=lambda x: x["accuracy"])

    # AI評価アドバイス（バイリンガル）
    weak_cats = get_weak_categories(history, top_n=5)
    cat_details = []
    for cat in weak_cats:
        cat_rows = [r for r in history if r.get("category") == cat]
        total = len(cat_rows)
        correct = sum(1 for r in cat_rows if r.get("is_correct"))
        accuracy = (correct / total * 100) if total > 0 else 0
        cat_details.append({
            "name": ALL_CATEGORIES.get(cat, cat),
            "accuracy": accuracy,
            "total": total
        })

    prompt = f"""以下の学習データに基づいて、学習者への評価シート用AIアドバイスを作成してください。

【学習データ】
- 総回答数: {len(history)}問
- 文法正解率: {grammar_acc:.1f}%（{grammar_correct}/{grammar_total}問）
- 単語正解率: {vocab_acc:.1f}%（{vocab_correct}/{vocab_total}問）
- 苦手な分野:
"""
    for d in cat_details:
        prompt += f"  - {d['name']}: 正解率 {d['accuracy']:.0f}%（{d['total']}問）\n"

    prompt += """
【出力条件】
以下の5セクションについて、それぞれ「インドネシア語」と「日本語」で同じ内容を出力してください。
形式は以下の通り：

```html
<div class="bilingual-section">
  <div class="id">
    <h3>🇮🇩 [セクション名（インドネシア語）]</h3>
    <p>[内容（インドネシア語）]</p>
  </div>
  <div class="ja">
    <h3>🇯🇵 [セクション名（日本語）]</h3>
    <p>[内容（日本語）]</p>
  </div>
</div>
```

【5セクション】
1. 総合評価（Penilaian Keseluruhan / 総合評価）
2. 強み（Kelebihan / 強み）
3. 改善点（Area yang Perlu Ditingkatkan / 改善点）
4. 次のステップ（Langkah Selanjutnya / 次のステップ）
5. 学習モチベーション（Semangat Belajar / 学習モチベーション）

内容は具体的に、データに基づいて作成してください。励ましの言葉を含めてください。
"""

    ai_advice = call_openai(prompt, temperature=0.7, max_tokens=3000)

    # HTML組み立て
    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>学習評価シート</title>
<style>
@page {{ size: A4 portrait; margin: 15mm; }}
body {{
    font-family: 'Noto Sans JP', 'Hiragino Sans', 'Yu Gothic', sans-serif;
    font-size: 11pt;
    line-height: 1.8;
    color: #2c2c2c;
    max-width: 190mm;
    margin: 0 auto;
    padding: 10mm;
}}
h1 {{
    text-align: center;
    color: #1b4d3e;
    border-bottom: 3px solid #1b4d3e;
    padding-bottom: 10px;
    margin-bottom: 20px;
}}
h2 {{
    color: #1b4d3e;
    border-left: 4px solid #1b4d3e;
    padding-left: 10px;
    margin-top: 24px;
}}
.summary-box {{
    background: #f5f5f5;
    border-radius: 10px;
    padding: 16px;
    margin: 14px 0;
}}
.summary-item {{
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid #e0e0e0;
}}
.bar-container {{
    background: #e0e0e0;
    border-radius: 10px;
    height: 24px;
    margin: 6px 0;
    position: relative;
    overflow: hidden;
}}
.bar-fill {{
    height: 100%;
    border-radius: 10px;
    text-align: right;
    padding-right: 8px;
    color: #fff;
    font-weight: bold;
    font-size: 10pt;
    line-height: 24px;
}}
.bar-fill.high {{ background: #4caf50; }}
.bar-fill.medium {{ background: #ff9800; }}
.bar-fill.low {{ background: #f44336; }}

.bilingual-section {{
    display: flex;
    gap: 12px;
    margin: 12px 0;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    overflow: hidden;
}}
.bilingual-section .id {{
    flex: 1;
    background: #e8f5e9;
    padding: 12px;
    border-right: 2px solid #c8e6c9;
}}
.bilingual-section .ja {{
    flex: 1;
    background: #e3f2fd;
    padding: 12px;
}}
.bilingual-section h3 {{
    font-size: 11pt;
    margin: 0 0 8px 0;
    color: #1b4d3e;
}}
.bilingual-section p {{
    font-size: 10pt;
    margin: 0;
}}

.schedule-table {{
    width: 100%;
    border-collapse: collapse;
    margin: 10px 0;
    font-size: 10pt;
}}
.schedule-table th {{
    background: #1b4d3e;
    color: #fff;
    padding: 8px;
    text-align: left;
}}
.schedule-table td {{
    padding: 8px;
    border-bottom: 1px solid #e0e0e0;
}}
.priority-high {{ color: #f44336; font-weight: bold; }}
.priority-medium {{ color: #ff9800; font-weight: bold; }}
.priority-low {{ color: #4caf50; font-weight: bold; }}

.print-btn {{
    display: block;
    width: 200px;
    margin: 20px auto;
    padding: 12px;
    background: #1b4d3e;
    color: #fff;
    border: none;
    border-radius: 8px;
    font-size: 12pt;
    cursor: pointer;
}}
@media print {{
    .print-btn {{ display: none; }}
    body {{ font-size: 10pt; }}
}}
</style>
</head>
<body>
<button class="print-btn" onclick="window.print()">🖨️ 印刷 / PDF保存</button>

<h1>📊 学習評価シート</h1>
<p style="text-align:center;color:#666;">生成日: {datetime.now().strftime('%Y年%m月%d日')}</p>

<h2>📋 学習サマリー</h2>
<div class="summary-box">
    <div class="summary-item"><span>総回答数</span><span>{len(history)} 問</span></div>
    <div class="summary-item"><span>文法正解率</span><span>{grammar_acc:.1f}%（{grammar_correct}/{grammar_total}問）</span></div>
    <div class="summary-item"><span>単語正解率</span><span>{vocab_acc:.1f}%（{vocab_correct}/{vocab_total}問）</span></div>
</div>

<h2>📈 カテゴリ別正解率</h2>
"""

    for cat, stats in sorted(cat_stats.items(), key=lambda x: x[1]["correct"]/x[1]["total"] if x[1]["total"]>0 else 0):
        acc = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
        bar_class = "high" if acc >= 85 else "medium" if acc >= 70 else "low"
        html += f"""
<div style="margin:8px 0;">
    <div style="display:flex;justify-content:space-between;font-size:10pt;">
        <span>{ALL_CATEGORIES.get(cat, cat)}</span>
        <span>{stats['correct']}/{stats['total']}問（{acc:.0f}%）</span>
    </div>
    <div class="bar-container">
        <div class="bar-fill {bar_class}" style="width:{acc}%;">{acc:.0f}%</div>
    </div>
</div>
"""

    html += f"""
<h2>🤖 AI評価とアドバイス</h2>
{ai_advice or "<p>AI評価の生成に失敗しました。</p>"}

<h2>📅 復習スケジュール</h2>
<table class="schedule-table">
    <tr><th>分野</th><th>正解率</th><th>次回復習</th><th>優先度</th></tr>
"""

    for item in review_schedule:
        p_class = f"priority-{item['priority']}"
        html += f"""
    <tr>
        <td>{item['category']}</td>
        <td>{item['accuracy']:.0f}%</td>
        <td>{item['next_review']}</td>
        <td class="{p_class}">{item['priority']}</td>
    </tr>
"""

    html += """
</table>

<footer style="margin-top:30px;text-align:center;font-size:8pt;color:#aaa;">
    日本語AI学習システム v6.0 | GPT-4o mini + SM-2
</footer>
</body>
</html>
"""

    return html


# ============================================================
# HTMLプリント用CSS
# ============================================================
WORKSHEET_CSS = """
<style>
@page { size: A4 portrait; margin: 15mm; }
body {
    font-family: 'Noto Sans JP', 'Hiragino Sans', 'Yu Gothic', sans-serif;
    font-size: 11pt;
    line-height: 1.8;
    color: #2c2c2c;
    max-width: 190mm;
    margin: 0 auto;
    padding: 10mm;
}
h1 {
    text-align: center;
    color: #1b4d3e;
    border-bottom: 3px solid #1b4d3e;
    padding-bottom: 10px;
}
h2 {
    color: #1b4d3e;
    border-left: 4px solid #1b4d3e;
    padding-left: 10px;
    margin-top: 24px;
}
ol li { margin-bottom: 14px; }
.answer-line {
    border-bottom: 1px solid #333;
    display: inline-block;
    width: 120px;
    margin-left: 8px;
}
.print-btn {
    display: block;
    width: 200px;
    margin: 20px auto;
    padding: 12px;
    background: #1b4d3e;
    color: #fff;
    border: none;
    border-radius: 8px;
    font-size: 12pt;
    cursor: pointer;
}
.bilingual-section {
    display: flex;
    gap: 12px;
    margin: 12px 0;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    overflow: hidden;
}
.bilingual-section .id {
    flex: 1;
    background: #e8f5e9;
    padding: 12px;
    border-right: 2px solid #c8e6c9;
}
.bilingual-section .ja {
    flex: 1;
    background: #e3f2fd;
    padding: 12px;
}
.bilingual-section h3 {
    font-size: 11pt;
    margin: 0 0 8px 0;
    color: #1b4d3e;
}
.bilingual-section p {
    font-size: 10pt;
    margin: 0;
}
@media print {
    .print-btn { display: none; }
    body { font-size: 10pt; }
}
</style>
"""


# ============================================================
# Streamlit UI
# ============================================================
def main():
    # session_state 初期化
    if "history" not in st.session_state:
        st.session_state.history = []
    if "test_questions" not in st.session_state:
        st.session_state.test_questions = []
    if "test_answers" not in st.session_state:
        st.session_state.test_answers = {}

    st.sidebar.title("🎌 メニュー")
    page = st.sidebar.radio(
        "ページを選択",
        ["📝 プリント生成", "🔄 復習テストプリント", "🧪 理解度テスト", "📊 学習履歴・復習スケジュール", "📈 評価シート生成"]
    )

    # ============================================================
    # ページ1: プリント生成
    # ============================================================
    if page == "📝 プリント生成":
        st.title("📝 プリント生成")
        st.markdown("文法と単語を統合した学習プリントをAIが生成します。")

        col1, col2 = st.columns(2)
        with col1:
            level = st.selectbox("学習レベル", ["初級", "中級", "上級"])
        with col2:
            theme = st.selectbox("テーマ", ["日常会話", "買い物", "レストラン", "仕事", "学校", "旅行", "自由"])

        num_questions = st.slider("問題数", 5, 20, 10)

        if st.button("🚀 プリントを生成", type="primary"):
            with st.spinner("AIがプリントを作成中...（約10〜20秒）"):
                worksheet = generate_worksheet(
                    level, theme, num_questions,
                    history=st.session_state.history
                )

            if worksheet:
                # HTML組み立て
                html_content = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>日本語学習プリント</title>
{WORKSHEET_CSS}
</head>
<body>
<button class="print-btn" onclick="window.print()">🖨️ 印刷 / PDF保存</button>
{worksheet}
<footer style="margin-top:30px;text-align:center;font-size:8pt;color:#aaa;">
    日本語AI学習システム v6.0 | GPT-4o mini + SM-2
</footer>
</body>
</html>"""

                st.success("✅ プリントが生成されました！")
                st.download_button(
                    label="📥 HTMLをダウンロード",
                    data=html_content.encode('utf-8'),
                    file_name=f"worksheet_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html"
                )
                st.components.v1.html(html_content, height=600, scrolling=True)

    # ============================================================
    # ページ2: 復習テストプリント
    # ============================================================
    elif page == "🔄 復習テストプリント":
        st.title("🔄 復習テストプリント")
        st.markdown("間違いが多い分野を集中的に復習するプリントを生成します。")

        if not st.session_state.history:
            st.warning("⚠️ まだ学習履歴がありません。先に「🧪 理解度テスト」を受けてください。")
        else:
            weak_cats = get_weak_categories(st.session_state.history)
            if weak_cats:
                st.info(f"📌 重点復習分野: {', '.join([ALL_CATEGORIES.get(c, c) for c in weak_cats[:5]])}")

            num_q = st.slider("問題数", 8, 15, 10)

            if st.button("🚀 復習プリントを生成", type="primary"):
                with st.spinner("AIが復習プリントを作成中...（約15〜30秒）"):
                    # AI評価アドバイスを生成
                    advice = generate_review_advice(st.session_state.history)
                    worksheet = generate_review_worksheet(st.session_state.history, num_q)

                if worksheet:
                    html_content = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>復習テストプリント</title>
{WORKSHEET_CSS}
</head>
<body>
<button class="print-btn" onclick="window.print()">🖨️ 印刷 / PDF保存</button>

<h1>🔄 復習テストプリント</h1>
<p style="text-align:center;color:#666;">生成日: {datetime.now().strftime('%Y年%m月%d日')}</p>

<h2>🤖 AI評価とアドバイス</h2>
{advice or "<p>AI評価の生成に失敗しました。</p>"}

<hr style="margin:24px 0;border:none;border-top:2px solid #1b4d3e;">

{worksheet}

<footer style="margin-top:30px;text-align:center;font-size:8pt;color:#aaa;">
    日本語AI学習システム v6.0 | GPT-4o mini + SM-2
</footer>
</body>
</html>"""

                    st.success("✅ 復習プリントが生成されました！")
                    st.download_button(
                        label="📥 HTMLをダウンロード",
                        data=html_content.encode('utf-8'),
                        file_name=f"review_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html"
                    )
                    st.components.v1.html(html_content, height=700, scrolling=True)

    # ============================================================
    # ページ3: 理解度テスト
    # ============================================================
    elif page == "🧪 理解度テスト":
        st.title("🧪 理解度テスト")
        st.markdown("ブラウザ上でテストを受け、結果を記録します。")

        test_type = st.selectbox("テスト種類", ["文法", "単語", "混合"])
        num_questions = st.slider("問題数", 3, 10, 5)

        if st.button("🚀 テスト開始", type="primary"):
            with st.spinner("AIが問題を作成中..."):
                if test_type == "文法":
                    cats = list(GRAMMAR_CATEGORIES.keys())
                elif test_type == "単語":
                    cats = list(VOCAB_CATEGORIES.keys())
                else:
                    cats = list(ALL_CATEGORIES.keys())

                selected_cats = random.sample(cats, min(num_questions, len(cats)))
                questions = generate_test_questions(selected_cats, num_questions)

            if questions:
                st.session_state.test_questions = questions
                st.session_state.test_answers = {}
                st.success(f"✅ {len(questions)}問のテストが準備できました！")
                st.rerun()

        if st.session_state.test_questions:
            st.markdown("---")
            st.subheader("📝 問題に回答してください")

            for i, q in enumerate(st.session_state.test_questions):
                st.markdown(f"**{i+1}. {q['question']}**")
                for choice_key, choice_text in q['choices'].items():
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;**{choice_key}**) {choice_text}")

                answer = st.radio(
                    f"第{i+1}問の答え",
                    options=list(q['choices'].keys()),
                    key=f"q_{i}",
                    horizontal=True,
                    label_visibility="collapsed"
                )
                st.session_state.test_answers[i] = answer
                st.markdown("")

            if st.button("✅ 採点する", type="primary"):
                results = []
                correct_count = 0

                for i, q in enumerate(st.session_state.test_questions):
                    user_ans = st.session_state.test_answers.get(i, "")
                    is_correct = user_ans == q['correct']
                    if is_correct:
                        correct_count += 1

                    # 品質スコア（SM-2用）
                    quality = 5 if is_correct else 1

                    # 履歴に追加
                    result_row = {
                        "test_id": f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}",
                        "question": q['question'],
                        "category": q.get('category', ''),
                        "question_type": test_type,
                        "is_correct": is_correct,
                        "quality": quality,
                        "reps": 0,
                        "ef": 2.5,
                        "interval": 0,
                        "answered_at": datetime.now().isoformat(),
                        "next_review_at": (datetime.now() + timedelta(days=1)).isoformat(),
                    }
                    results.append(result_row)
                    st.session_state.history.append(result_row)

                # 結果表示
                st.markdown("---")
                st.subheader("📊 採点結果")
                accuracy = (correct_count / len(st.session_state.test_questions) * 100)
                st.metric("正解率", f"{correct_count}/{len(st.session_state.test_questions)}問", f"{accuracy:.0f}%")

                for i, q in enumerate(st.session_state.test_questions):
                    user_ans = st.session_state.test_answers.get(i, "")
                    is_correct = user_ans == q['correct']
                    icon = "✅" if is_correct else "❌"
                    st.markdown(f"{icon} **第{i+1}問**: あなたの答え **{user_ans}** | 正解 **{q['correct']}**")
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;💡 {q.get('explanation', '解説なし')}")

                if accuracy >= 80:
                    st.success("🎉 素晴らしい！よく理解できています。")
                elif accuracy >= 60:
                    st.warning("👍 もう少し復習しましょう。")
                else:
                    st.error("📚 もう一度学習してからテストを受け直しましょう。")

                # 履歴ダウンロード
                if st.session_state.history:
                    csv_buffer = io.StringIO()
                    writer = csv.DictWriter(csv_buffer, fieldnames=st.session_state.history[0].keys())
                    writer.writeheader()
                    writer.writerows(st.session_state.history)
                    st.download_button(
                        label="📥 学習履歴をCSVダウンロード",
                        data=csv_buffer.getvalue().encode('utf-8'),
                        file_name=f"history_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )

    # ============================================================
    # ページ4: 学習履歴・復習スケジュール
    # ============================================================
    elif page == "📊 学習履歴・復習スケジュール":
        st.title("📊 学習履歴・復習スケジュール")

        if not st.session_state.history:
            st.info("まだ学習履歴がありません。「🧪 理解度テスト」を受けてください。")
        else:
            # カテゴリ別正解率
            st.subheader("📈 カテゴリ別正解率")
            cat_stats = defaultdict(lambda: {"correct": 0, "total": 0})
            for row in st.session_state.history:
                cat = row.get("category", "")
                if cat:
                    cat_stats[cat]["total"] += 1
                    if row.get("is_correct"):
                        cat_stats[cat]["correct"] += 1

            for cat, stats in sorted(cat_stats.items(), key=lambda x: x[1]["correct"]/x[1]["total"] if x[1]["total"]>0 else 0):
                acc = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
                bar_color = "#4caf50" if acc >= 85 else "#ff9800" if acc >= 70 else "#f44336"
                st.markdown(f"**{ALL_CATEGORIES.get(cat, cat)}**: {stats['correct']}/{stats['total']}問（{acc:.0f}%）")
                st.progress(acc / 100, text=f"{acc:.0f}%")

            # 復習スケジュール
            st.subheader("📅 復習スケジュール")
            today = datetime.now().date()
            schedule_data = []
            for cat, stats in cat_stats.items():
                if stats["total"] >= 2:
                    acc = stats["correct"] / stats["total"]
                    if acc < 0.7:
                        schedule_data.append({"分野": ALL_CATEGORIES.get(cat, cat), "正解率": f"{acc*100:.0f}%", "次回復習": "明日", "優先度": "🔴 高"})
                    elif acc < 0.85:
                        schedule_data.append({"分野": ALL_CATEGORIES.get(cat, cat), "正解率": f"{acc*100:.0f}%", "次回復習": "3日後", "優先度": "🟡 中"})
                    else:
                        schedule_data.append({"分野": ALL_CATEGORIES.get(cat, cat), "正解率": f"{acc*100:.0f}%", "次回復習": "1週間後", "優先度": "🟢 低"})

            if schedule_data:
                st.table(sorted(schedule_data, key=lambda x: x["正解率"]))

            # 回答履歴
            st.subheader("📝 最近の回答履歴（最新20件）")
            recent = st.session_state.history[-20:][::-1]
            for row in recent:
                icon = "✅" if row.get("is_correct") else "❌"
                cat_name = ALL_CATEGORIES.get(row.get("category", ""), row.get("category", ""))
                st.markdown(f"{icon} **{cat_name}** | {row.get('question', '')[:40]}... | {row.get('answered_at', '')[:10]}")

        # 履歴アップロー
        st.subheader("📤 履歴のアップロード")
        uploaded = st.file_uploader("以前ダウンロードしたCSVをアップロード", type="csv")
        if uploaded:
            import pandas as pd
            df = pd.read_csv(uploaded)
            st.session_state.history = df.to_dict('records')
            st.success("✅ 履歴を読み込みました！")

    # ============================================================
    # ページ5: 評価シート生成
    # ============================================================
    elif page == "📈 評価シート生成":
        st.title("📈 評価シート生成")
        st.markdown("学習履歴を基に、AIが評価シートを自動生成します。")

        if st.button("🚀 評価シートを生成", type="primary"):
            with st.spinner("AIが評価シートを作成中...（約20〜40秒）"):
                sheet_html = generate_evaluation_sheet(st.session_state.history)

            if sheet_html:
                st.success("✅ 評価シートが生成されました！")
                st.download_button(
                    label="📥 HTMLをダウンロード",
                    data=sheet_html.encode('utf-8'),
                    file_name=f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html"
                )
                st.components.v1.html(sheet_html, height=700, scrolling=True)


if __name__ == "__main__":
    main()
