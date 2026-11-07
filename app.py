"""Sideline web app. Run locally: streamlit run app.py"""
import json
import os
from pathlib import Path

import streamlit as st

from rag.config import DEFAULT_MODELS, load_settings
from sports import db
from sports.assistant import build_assistant, load_follows

KEYS = {"gemini": "GEMINI_API_KEY", "claude": "ANTHROPIC_API_KEY"}
EXAMPLES = [
    "Who leads college football in rushing touchdowns?",
    "How did Northwestern do last weekend?",
    "Who was the top goal scorer in women's lacrosse last season?",
    "How are my players doing?",
]

st.set_page_config(page_title="Sideline", page_icon="🏟️")
settings = load_settings()


@st.cache_resource
def get_assistant(provider: str):
    return build_assistant(load_settings(provider=provider))


def available_providers() -> list:
    providers = [p for p, key in KEYS.items() if os.getenv(key)]
    if os.getenv("RAG_ENABLE_OLLAMA"):
        providers.append("ollama")
    return providers or [settings.provider]


# ---------- sidebar: favourites and data status ----------
with st.sidebar:
    st.subheader("My players & teams")
    follows = load_follows(settings.follows_file)
    players = st.text_area("Players (one per line)", "\n".join(follows.get("players", [])))
    teams = st.text_area("Teams (one per line)", "\n".join(follows.get("teams", [])))
    if st.button("Save favourites"):
        Path(settings.follows_file).write_text(json.dumps({
            "players": [p.strip() for p in players.splitlines() if p.strip()],
            "teams": [t.strip() for t in teams.splitlines() if t.strip()],
        }, indent=2))
        st.success("Saved")
    st.divider()
    st.caption(f"Stats updated: {db.get_meta(settings.sports_db, 'last_update') or 'not yet, run the update'}")
    providers = available_providers()
    provider = st.selectbox(
        "Model", providers,
        format_func=lambda p: f"{p} ({os.getenv('RAG_MODEL') or DEFAULT_MODELS[p]})",
    )

# ---------- main ----------
st.title("🏟️ Sideline")
st.markdown("**In college it's hard to keep up.** Ask about your teams and favourite players "
            "and get the answer in seconds, with the stats and stories it came from.")

choice = st.radio("Try one:", EXAMPLES, index=None, horizontal=False, label_visibility="collapsed")
question = st.text_input("Ask anything about college football or lacrosse", value=choice or "")

if question:
    with st.spinner("Checking the stats..."):
        answer = get_assistant(provider).answer(question)
    st.markdown(f"### {answer.text}")
    with st.expander(f"How I found this · {answer.latency_s:.1f}s · {answer.provider}/{answer.model}"):
        if answer.sql:
            st.caption("Stats query")
            st.code(answer.sql, language="sql")
            if answer.sql_rows:
                st.dataframe(answer.sql_rows, use_container_width=True)
        if answer.sql_error:
            st.warning(f"Query failed: {answer.sql_error}")
        if answer.sources:
            st.caption(f"News searched for: {answer.search}")
            for number, source in enumerate(answer.sources, start=1):
                st.write(f"[{number}] {source['source']} (similarity {source['score']})")
