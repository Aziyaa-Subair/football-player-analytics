"""
Football Scouting Tool
Find players with a similar playing style, optionally within a budget.
Data: players.csv, exported from football_player_analytics.ipynb
(per-90 stats + KMeans cluster for 1,104 outfield players, synthetic dataset).
"""
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

STAT_COLS = ['goals', 'assists', 'shots', 'shots_on_target', 'expected_goals_xg',
             'expected_assists_xa', 'key_passes', 'successful_passes', 'total_passes',
             'dribbles_attempted', 'successful_dribbles', 'crosses', 'tackles',
             'interceptions', 'clearances', 'blocks', 'aerial_duels_won', 'recoveries']

st.set_page_config(page_title='Football Scouting Tool', page_icon='⚽', layout='wide')


# ---------- data (cached: runs once, not on every click) ----------
@st.cache_data
def load_data():
    players = pd.read_csv('players.csv', index_col='player_id')

    # same steps as the notebook: scale, then cosine similarity
    X = StandardScaler().fit_transform(players[STAT_COLS])
    sim = pd.DataFrame(cosine_similarity(X), index=players.index, columns=players.index)
    z = pd.DataFrame(X, index=players.index, columns=STAT_COLS)   # scaled stats, for the style chart

    # name each cluster after the position most of its players have
    cluster_names = players.groupby('cluster')['position'].agg(lambda s: s.mode()[0] + 's')
    players['style_group'] = players['cluster'].map(cluster_names)

    # names are not unique, so the dropdown shows name + team
    players['label'] = players['player_name'] + ' (' + players['team'] + ', ' + players['position'] + ')'
    return players, sim, z


def find_similar(players, sim, pid, n=5, max_value=None):
    scores = sim[pid].drop(pid)
    result = players.loc[scores.index, ['player_name', 'team', 'position', 'market_value_eur']].copy()
    result['similarity'] = scores.round(3)
    if max_value is not None:
        result = result[result['market_value_eur'] <= max_value]
    return result.sort_values('similarity', ascending=False).head(n)


players, sim, z = load_data()

# ---------- sidebar: inputs ----------
st.sidebar.header('Search')
labels = players['label'].sort_values()
choice = st.sidebar.selectbox('Player', labels.index, format_func=lambda pid: players.loc[pid, 'label'])
n = st.sidebar.slider('Number of results', 3, 15, 5)
use_budget = st.sidebar.checkbox('Limit by budget')
budget_m = st.sidebar.slider('Max market value (€M)', 0.5, 100.0, 5.0, step=0.5, disabled=not use_budget)
max_value = budget_m * 1_000_000 if use_budget else None

# ---------- main page ----------
st.title('⚽ Football Scouting Tool')
st.caption('Find players who play like a chosen player, using cosine similarity on 18 per-90 stats. '
           'Data is synthetic (Kaggle "FIFA World Cup 2026 Player Performance").')

tab_scout, tab_groups = st.tabs(['Similar players', 'Style groups'])

with tab_scout:
    p = players.loc[choice]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Player', p['player_name'])
    c2.metric('Team', p['team'])
    c3.metric('Position', p['position'])
    c4.metric('Market value', f"€{p['market_value_eur'] / 1e6:.1f}M")

    result = find_similar(players, sim, choice, n=n, max_value=max_value)

    st.subheader('Most similar players')
    if result.empty:
        st.warning('No players found within this budget. Try raising it.')
    else:
        table = result.rename(columns={'player_name': 'Player', 'team': 'Team', 'position': 'Position',
                                       'market_value_eur': 'Value (€M)', 'similarity': 'Similarity'})
        table['Value (€M)'] = (table['Value (€M)'] / 1e6).round(2)
        st.dataframe(table, width='stretch', hide_index=True)

        cheaper = (result['market_value_eur'] < p['market_value_eur']).sum()
        st.caption(f'{cheaper} of {len(result)} are cheaper than {p["player_name"]}.')

        # style comparison: chosen player vs the top match (scaled: 0 = average player)
        top_pid = result.index[0]
        st.subheader(f'Style: {p["player_name"]} vs {players.loc[top_pid, "player_name"]}')
        st.caption('Scaled per-90 stats: 0 = average outfield player, +1 = one standard deviation above.')
        compare = pd.DataFrame({p['player_name']: z.loc[choice],
                                players.loc[top_pid, 'player_name'] + ' ': z.loc[top_pid]})
        st.bar_chart(compare, horizontal=True, stack=False, height=500)

with tab_groups:
    st.subheader('Playing-style groups (KMeans, k = 3)')
    st.caption('Position was not an input to the clustering; groups are named after their most common position.')
    profile = players.groupby('style_group')[STAT_COLS].mean().round(2).T
    st.dataframe(profile, width='stretch')
    st.write(players['style_group'].value_counts().rename('players'))
