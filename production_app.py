import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import pandas as pd
import plotly.express as px
from collections import defaultdict
from pathlib import Path
import time
import traceback

# -------------------------------
# Set Page Configuration
# -------------------------------
st.set_page_config(
    page_title="LoEJ Perfect Parlay",
    page_icon="🎲",
    layout="wide"
)

# -------------------------------
# Firebase Initialization
# -------------------------------
def initialize_firebase():
    if 'firebase_initialized' not in st.session_state:
        try:
            if not firebase_admin._apps:
                cred = credentials.Certificate(dict(st.secrets["firebase"]))
                firebase_admin.initialize_app(cred)
            st.session_state['firebase_initialized'] = True
        except Exception as e:
            st.error(f"Failed to initialize Firebase: {str(e)}")
            return None
    return firestore.client()

db = initialize_firebase()
if db is None:
    st.stop()

# -------------------------------
# ParlayData Class
# -------------------------------
class ParlayData:
    DEFAULT_SETTINGS = {
        "owners": ["James", "Jon", "Matt", "Pat", "Jack", "Jeff", "Sam/Connor", "Enzo", "Dillon", "Cole", "TBD"],
        "parlay_fee": 20,
        "parlay_president": "James Quaglia"
    }

    def __init__(self, db):
        self.db = db
        self.settings = self.get_settings()

    def get_settings(self):
        settings_ref = self.db.collection('settings').document('default')
        settings_doc = settings_ref.get()
        if settings_doc.exists:
            return settings_doc.to_dict()
        else:
            try:
                settings_ref.set(self.DEFAULT_SETTINGS)
                st.info("Default settings initialized in Firestore.")
                return self.DEFAULT_SETTINGS
            except Exception as e:
                st.error(f"Failed to set default settings: {e}")
                return self.DEFAULT_SETTINGS

    def get_week_key(self, week, year):
        return f"Week {week} {year}"

    def get_week_bets(self, week, year):
        week_key = self.get_week_key(week, year)
        week_doc = self.db.collection('parlay_history').document(week_key).get()

        if week_doc.exists:
            week_data = week_doc.to_dict()
            if week_data:
                week_data['doc_id'] = week_doc.id
                return [week_data]
        return []

    def add_bet(self, week, year, parlay_info, bet):
        week_key = self.get_week_key(week, year)
        week_ref = self.db.collection('parlay_history').document(week_key)
        try:
            week_ref.set({
                "owner": parlay_info["owner"],
                "date": parlay_info["date"],
                "total_odds": parlay_info["total_odds"],
                "parlay_result": "PENDING",
                "bets": firestore.ArrayUnion([bet])
            }, merge=True)
            st.success(f"Bet added to {week_key}.")
        except Exception as e:
            st.error(f"Failed to add bet: {e}")

    def save_results_from_doc_id(self, doc_id, bets):
        parlay_ref = self.db.collection('parlay_history').document(doc_id)
        try:
            parlay_ref.update({"bets": bets})
            st.success("Results saved successfully.")
        except Exception as e:
            st.error(f"Failed to save results: {e}")

    def delete_bet_from_doc_id(self, doc_id, bet_index):
        try:
            parlay_ref = self.db.collection('parlay_history').document(doc_id)
            doc = parlay_ref.get()

            if not doc.exists:
                st.error("Parlay not found.")
                return False

            current_data = doc.to_dict()
            current_bets = current_data.get('bets', [])

            if 0 <= bet_index < len(current_bets):
                del current_bets[bet_index]

                if current_bets:
                    parlay_ref.update({'bets': current_bets})
                    st.success("Bet deleted successfully!")
                else:
                    parlay_ref.delete()
                    st.success(f"Document '{doc_id}' removed as no bets remain.")
                return True
            else:
                st.error(f"Invalid bet index: {bet_index}")
                return False

        except Exception as e:
            st.error(f"Error in delete_bet_from_doc_id: {str(e)}")
            return False

# -------------------------------
# BetAnalyzer Class
# -------------------------------
class BetAnalyzer:
    @staticmethod
    def calculate_metrics(parlays):
        metrics = defaultdict(lambda: {"win": 0, "loss": 0, "pending": 0})
        for parlay in parlays:
            for bet in parlay.get("bets", []):
                owner = bet.get("owner")
                result = bet.get("result", "").lower()
                if owner != "TBD" and result in metrics[owner]:
                    metrics[owner][result] += 1
        return metrics

# -------------------------------
# ParlayUI Class
# -------------------------------
class ParlayUI:
    def __init__(self, db):
        self.data = ParlayData(db)
        self.analyzer = BetAnalyzer()

    def setup_page(self):
        logo_path = "logo.png"
        if Path(logo_path).exists():
            st.sidebar.image(logo_path, use_container_width=True)

        st.title("League of Extraordinary Jabrones")
        st.subheader("Weekly Low Scorer Parlay Tracker")

        pages = {
            "Submit Parlay": self.show_submission_form,
            "View History": self.show_history,
            "Manage Results": self.manage_results,
            "Statistics": self.show_statistics,
            "Owner Stats": self.show_owner_stats,
            "Leaderboard": self.show_leaderboard,
            "Reset History": self.reset_history
        }

        page = st.sidebar.selectbox("Navigation", list(pages.keys()))
        pages[page]()

    def show_submission_form(self):
        st.header("Submit Weekly Parlay")

        col1, col2 = st.columns(2)
        with col1:
            week = st.number_input("NFL Week", 1, 18, 1, key="submit_week")
            year = st.number_input("Year", 2024, 2030, 2024, key="submit_year")
        with col2:
            submitter = st.selectbox("Submitter", self.data.settings["owners"], key="submit_owner")
            date = st.date_input("Date", datetime.today(), key="submit_date")

        week_key = self.data.get_week_key(week, year)
        existing_bets = self.data.get_week_bets(week, year)
        submitted_owners = {bet["owner"] for parlay in existing_bets for bet in parlay.get("bets", []) if bet["owner"] != "TBD"}

        available_owners = [o for o in self.data.settings["owners"] if o not in submitted_owners]

        st.markdown("""
        ### Bet Format Guidelines:
        - **Team vs Team Over/Under X.Y**
          - *Example:* `Under 48.5 Cardinals vs Dolphins`
        - **Anytime TD Scorer - Player Team @ Opponent**
          - *Example:* `Anytime TD Scorer - WR Cardinals @ NE Patriots`
        - **Spread Bets with SGP (Same Game Parlay)**
          - *Example:* `+5 3Pks SGP TEN Titans @ DET Lions`
        - **Multiple Picks with SGP**
          - *Example:* `2 Picks SGP KC Chiefs @ LA Chargers`
        - **Other formats should clearly state the type and details of the bet.**
        """)

        bet_description = st.text_input("Your Bet", key="submit_bet")
        owner = st.selectbox("Select Owner", ["TBD"] + available_owners, key="submit_bet_owner")

        if st.button("Submit Bet", key="submit_bet_button"):
            if bet_description and owner != "TBD":
                parlay_info = {
                    "owner": submitter,
                    "date": date.strftime("%Y-%m-%d"),
                    "total_odds": "+0",
                    "parlay_result": "PENDING"
                }
                new_bet = {
                    "bet": bet_description,
                    "result": "PENDING",
                    "owner": owner
                }

                self.data.add_bet(week, year, parlay_info, new_bet)
                st.success(f"Bet submitted for {owner}!")
            else:
                st.error("Please enter a valid bet and select an owner.")

    def show_owner_stats(self):
        st.header("Owner Statistics")

        parlays = [doc.to_dict() for doc in self.data.db.collection('parlay_history').stream()]

        owner_stats = defaultdict(lambda: {"total_bets": 0, "win": 0, "loss": 0, "pending": 0})

        for parlay in parlays:
            for bet in parlay.get("bets", []):
                owner = bet.get("owner")
                result = bet.get("result", "").lower()
                if owner != "TBD" and result in ["win", "loss", "pending"]:
                    owner_stats[owner]["total_bets"] += 1
                    owner_stats[owner][result] += 1

        stats_df = pd.DataFrame([
            {
                "Owner": owner,
                "Total Bets": stats["total_bets"],
                "Wins": stats["win"],
                "Losses": stats["loss"],
                "Pending": stats["pending"],
                "Win Rate": f"{(stats['win'] / (stats['win'] + stats['loss']) * 100):.1f}%"
                if (stats['win'] + stats['loss']) > 0 else "0%"
            }
            for owner, stats in owner_stats.items()
        ])

        st.dataframe(stats_df)

        fig = px.bar(
            stats_df,
            x="Owner",
            y=["Wins", "Losses", "Pending"],
            title="Owner Performance",
            barmode="stack"
        )
        st.plotly_chart(fig)

    def show_leaderboard(self):
        st.header("Leaderboard")

        parlays = [doc.to_dict() for doc in self.data.db.collection('parlay_history').stream()]

        metrics = self.analyzer.calculate_metrics(parlays)

        leaderboard = pd.DataFrame([
            {
                "Owner": owner,
                "Wins": stats["win"],
                "Losses": stats["loss"],
                "Win Rate": (stats["win"] / (stats["win"] + stats["loss"]) * 100)
                if (stats["win"] + stats["loss"]) > 0 else 0,
                "Total Bets": stats["win"] + stats["loss"] + stats["pending"]
            }
            for owner, stats in metrics.items() if owner != "TBD"
        ])

        leaderboard = leaderboard.sort_values("Win Rate", ascending=False)
        leaderboard["Win Rate"] = leaderboard["Win Rate"].apply(lambda x: f"{x:.1f}%")

        st.dataframe(leaderboard)

    def manage_results(self):
        st.header("Manage Results")

        weeks = sorted([doc.id for doc in self.data.db.collection('parlay_history').stream()])

        if not weeks:
            st.warning("No parlays to manage")
            return

        selected_week = st.selectbox("Select Week", weeks, key="manage_results_week")

        week_ref = self.data.db.collection('parlay_history').document(selected_week)
        parlay = week_ref.get().to_dict()

        if parlay:
            st.subheader(f"Managing Results for {selected_week}")
            for i, bet in enumerate(parlay.get("bets", [])):
                st.markdown(f"**Bet {i+1}**: {bet.get('bet', '')} ({bet.get('owner', '')})")
                new_result = st.selectbox(
                    f"Result for {bet.get('owner', '')}'s bet",
                    ["PENDING", "WIN", "LOSS"],
                    index=["PENDING", "WIN", "LOSS"].index(bet.get("result", "PENDING").upper()),
                    key=f"manage_result_{selected_week}_{i}"
                )
                parlay["bets"][i]["result"] = new_result

            if st.button("Save Results", key=f"save_results_{selected_week}"):
                self.data.save_results_from_doc_id(selected_week, parlay["bets"])
        else:
            st.error("Selected week not found.")

    def show_statistics(self):
        st.header("Overall Statistics")

        parlays = [doc.to_dict() for doc in self.data.db.collection('parlay_history').stream()]

        total_bets = 0
        wins = 0
        losses = 0
        pending = 0

        for parlay in parlays:
            for bet in parlay.get("bets", []):
                owner = bet.get("owner")
                result = bet.get("result", "").lower()
                if owner != "TBD":
                    total_bets += 1
                    if result == "win":
                        wins += 1
                    elif result == "loss":
                        losses += 1
                    else:
                        pending += 1

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Bets", total_bets)
        with col2:
            win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
            st.metric("Win Rate", f"{win_rate:.1f}%")
        with col3:
            st.metric("Pending Bets", pending)

    def delete_bet(self, doc_id, bet_index):
        if 'delete_confirmation_state' not in st.session_state:
            st.session_state.delete_confirmation_state = {}

        deletion_key = f"delete_{doc_id}_{bet_index}"

        if deletion_key not in st.session_state.delete_confirmation_state:
            st.session_state.delete_confirmation_state[deletion_key] = "pending"

        try:
            parlay_ref = self.data.db.collection('parlay_history').document(doc_id)
            parlay = parlay_ref.get().to_dict()

            if not parlay or 'bets' not in parlay or bet_index >= len(parlay['bets']):
                st.error("Invalid parlay or bet index")
                return

            bet = parlay['bets'][bet_index]

            if st.session_state.delete_confirmation_state[deletion_key] == "pending":
                st.warning(f"Are you sure you want to delete this bet?")
                st.write(f"**Bet Details:**")
                st.write(f"- Owner: {bet['owner']}")
                st.write(f"- Bet: {bet['bet']}")
                st.write(f"- Result: {bet['result']}")

                col1, col2 = st.columns([1, 1])

                with col1:
                    if st.button("✔️ Confirm Delete", key=f"confirm_{deletion_key}"):
                        st.session_state.delete_confirmation_state[deletion_key] = "confirmed"
                        st.experimental_rerun()

                with col2:
                    if st.button("❌ Cancel", key=f"cancel_{deletion_key}"):
                        st.session_state.delete_confirmation_state[deletion_key] = "cancelled"
                        st.experimental_rerun()

            elif st.session_state.delete_confirmation_state[deletion_key] == "confirmed":
                if self.data.delete_bet_from_doc_id(doc_id, bet_index):
                    del st.session_state.delete_confirmation_state[deletion_key]
                    st.experimental_rerun()
            elif st.session_state.delete_confirmation_state[deletion_key] == "cancelled":
                del st.session_state.delete_confirmation_state[deletion_key]
                st.experimental_rerun()

        except Exception as e:
            st.error(f"Error in delete_bet UI handler: {str(e)}")

    def show_history(self):
        st.header("Parlay History")

        view = st.radio("View", ["By Week", "All Time"], key="history_view")

        if view == "By Week":
            weeks = sorted([doc.id for doc in self.data.db.collection('parlay_history').stream()])
            if not weeks:
                st.info("No bets found.")
                return
            selected_week = st.selectbox("Select Week", weeks, key="history_select_week")
            week_doc = self.data.db.collection('parlay_history').document(selected_week).get()
            parlays = [week_doc.to_dict()] if week_doc.exists else []
            if week_doc.exists:
                parlays[0]['doc_id'] = week_doc.id
        else:
            parlays = []
            for doc in self.data.db.collection('parlay_history').stream():
                parlay = doc.to_dict()
                parlay['doc_id'] = doc.id
                parlays.append(parlay)

        if not parlays:
            st.info("No bets found.")
            return

        for parlay in parlays:
            if parlay:
                st.subheader(f"Parlay by {parlay.get('owner', 'Unknown')} on {parlay.get('date', 'No date')}")

                bets = parlay.get('bets', [])
                for bet_index, bet in enumerate(bets):
                    if bet.get('owner') == "TBD":
                        continue

                    with st.expander(f"Bet {bet_index + 1}: {bet.get('bet', 'No bet description')}"):
                        col1, col2 = st.columns([3, 1])

                        with col1:
                            st.write(f"**Owner:** {bet.get('owner', 'Unknown')}")
                            st.write(f"**Result:** {bet.get('result', 'PENDING')}")

                        with col2:
                            doc_id = parlay.get('doc_id', f"parlay_{parlays.index(parlay)}")
                            if st.button("Delete", key=f"delete_button_{doc_id}_{bet_index}"):
                                self.delete_bet(doc_id, bet_index)

    def reset_history(self):
        st.header("Reset History")

        st.warning("This action will erase all submitted bets and cannot be undone.")
        if st.button("Reset All Bets", key="reset_all_bets"):
            confirm = st.checkbox("I confirm that I want to reset all history.", key="confirm_reset_history")
            if confirm:
                try:
                    for doc in self.data.db.collection('parlay_history').stream():
                        self.data.db.collection('parlay_history').document(doc.id).delete()
                    st.success("All bets have been reset. History is now empty.")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Failed to reset history: {e}")

    def run(self):
        self.setup_page()

# -------------------------------
# Main Application
# -------------------------------
if __name__ == "__main__":
    app = ParlayUI(db)
    app.run()
