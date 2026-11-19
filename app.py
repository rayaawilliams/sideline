"""Sideline web app. Run locally: streamlit run app.py"""
import json
import os
from pathlib import Path

import streamlit as st

from rag.config import DEFAULT_MODELS, load_settings
from sports import db
from sports.assistant import build_assistant, load_follows

KEYS = {"gemini": "GEMINI_API_KEY", "claude": "ANTHROPIC_API_KEY"}

st.set_page_config(page_title="Sideline", page_icon="🏟️")
settings = load_settings()


@st.cache_resource(show_spinner=False)
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
    st.write("**Teams:** " + (", ".join(follows.get("teams", [])) or "none yet"))
    st.write("**Players:** " + (", ".join(follows.get("players", [])) or "none yet"))
    if st.button("Edit favourites"):
        st.session_state.onboarded = False
    if st.button("Start over for a new fan"):
        Path(settings.follows_file).write_text(json.dumps({"players": [], "teams": []}, indent=2))
        st.session_state.onboarded = False
        st.session_state.chat = []
        st.rerun()
    st.divider()
    st.caption(f"Stats updated: {db.get_meta(settings.sports_db, 'last_update') or 'not yet, run the update'}")
    providers = available_providers()
    provider = st.selectbox(
        "Model", providers,
        format_func=lambda p: f"{p} ({os.getenv('RAG_MODEL') or DEFAULT_MODELS[p]})",
    )

# ---------- main ----------
st.title("🏟️ Sideline")
st.markdown("**Built for Northwestern students.** In college it's hard to keep up. Ask about the 'Cats, "
            "your favourite teams and players, and get the answer in seconds, with the stats and stories "
            "it came from.")


def show_work(answer) -> None:
    """The 'How I found this' panel under an answer."""
    with st.expander("How I found this"):
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


if "ready" not in st.session_state:
    # Load the models once, quietly, before the first question.
    with st.spinner("Getting Sideline ready..."):
        try:
            get_assistant(provider)
        except Exception:
            pass  # a missing key or busy model is reported when a question is asked
    st.session_state.ready = True

if "chat" not in st.session_state:
    st.session_state.chat = []  # list of (question, answer)


def make_it_yours() -> None:
    """Ask for favourite teams and players, picked from the ones Sideline has data for."""
    try:
        team_options, player_rows = db.follow_options(settings.sports_db)
    except Exception:
        st.info("No sports data yet: run `python -m sports.update` first.")
        return
    label = {f"{name} ({team})": name for name, team in player_rows}
    current = load_follows(settings.follows_file)
    with st.form("make_it_yours"):
        st.markdown("#### 👋 Make Sideline yours")
        st.caption("Pick 2-3 teams and players you follow. Start typing to search.")
        teams = st.multiselect("Favourite teams", team_options,
                               default=[t for t in current.get("teams", []) if t in team_options])
        players = st.multiselect("Favourite players", list(label),
                                 default=[k for k, v in label.items() if v in current.get("players", [])])
        if st.form_submit_button("Build my cheat sheet", type="primary"):
            if not teams and not players:
                st.warning("Pick at least one team or player.")
                return
            Path(settings.follows_file).write_text(json.dumps(
                {"players": [label[p] for p in players], "teams": teams}, indent=2))
            st.session_state.onboarded = True
            st.session_state.just_saved = True
            st.rerun()


if not st.session_state.get("onboarded"):
    make_it_yours()
elif st.session_state.pop("just_saved", False):
    saved = load_follows(settings.follows_file)
    st.success("Your cheat sheet is ready: " + ", ".join(saved.get("teams", []) + saved.get("players", []))
               + ". Ask away below.")

for past_question, past_answer in st.session_state.chat:
    with st.chat_message("user"):
        st.write(past_question)
    with st.chat_message("assistant", avatar="🏟️"):
        st.write(past_answer.text)
        show_work(past_answer)

question = st.chat_input("Ask about a team, a player or last weekend's game...")
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant", avatar="🏟️"):
        try:
            with st.spinner("Checking the stats..."):
                answer = get_assistant(provider).answer(question)
        except Exception as error:  # keep the app up during a live demo
            st.error("Sideline couldn't reach the AI model just now. Please try again in a moment.")
            with st.expander("Details"):
                st.code(str(error))
            st.stop()
        st.write(answer.text)
        show_work(answer)
    st.session_state.chat.append((question, answer))
