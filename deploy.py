import streamlit as st
import json
from datetime import datetime
import pandas as pd
import plotly.express as px
from collections import defaultdict
from pathlib import Path

class ParlayData:
    def __init__(self):
        self._initialize_session_state()
        
    def _initialize_session_state(self):
        """Initialize or load data from Streamlit secrets"""
        if 'data' not in st.session_state:
            try:
                st.session_state.data = self._load_from_secrets()
            except Exception:
                st.session_state.data = self.get_default_data()
    
    @staticmethod
    def _load_from_secrets():
        """Load data from Streamlit secrets"""
        try:
            parlay_history = json.loads(st.secrets["parlay_history"])
            settings = json.loads(st.secrets["settings"])
            return {
                "parlay_history": parlay_history,
                "settings": settings
            }
        except Exception as e:
            st.error(f"Error loading from secrets: {str(e)}")
            return ParlayData.get_default_data()
    
    @staticmethod
    def get_default_data():
        return {
            "parlay_history": {},
            "settings": {
                "owners": ["James", "Jon", "Matt", "Pat", "Jack", "Jeff", "Sam/Connor", "Enzo", "Dillon", "Cole", "TBD"],
                "parlay_fee": 20,
                "parlay_president": "James Quaglia"
            }
        }
    
    def save(self):
        """Save data to Streamlit secrets"""
        try:
            # Convert data to strings for secrets storage
            st.secrets["parlay_history"] = json.dumps(st.session_state.data["parlay_history"])
            st.secrets["settings"] = json.dumps(st.session_state.data["settings"])
            
            # Optional: Create a backup in secrets
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_key = f"backup_{timestamp}"
            st.secrets[backup_key] = json.dumps(st.session_state.data)
            
            # Keep only the last 5 backups
            self._cleanup_old_backups()
        except Exception as e:
            st.error(f"Error saving to secrets: {str(e)}")
    
    def _cleanup_old_backups(self):
        """Keep only the most recent 5 backups"""
        backup_keys = [k for k in st.secrets.keys() if k.startswith("backup_")]
        backup_keys.sort(reverse=True)
        
        for old_backup in backup_keys[5:]:
            del st.secrets[old_backup]
    
    def get_week_key(self, week, year):
        return f"Week {week} {year}"
    
    def get_week_bets(self, week, year):
        week_key = self.get_week_key(week, year)
        return st.session_state.data["parlay_history"].get(week_key, [])
    
    def get_owner_bet_for_week(self, owner, week, year):
        week_bets = self.get_week_bets(week, year)
        for parlay in week_bets:
            for bet in parlay["bets"]:
                if bet["owner"] == owner:
                    return bet
        return None

class BetAnalyzer:
    @staticmethod
    def classify_bet(bet_text):
        bet_text = bet_text.lower()
        if any(x in bet_text for x in ["td scorer", "touchdown"]): return "Touchdown Props"
        if any(x in bet_text for x in ["under", "over"]): return "Totals"
        if "sgp" in bet_text: return "Same Game Parlay"
        if "passing" in bet_text: return "Passing Props"
        if "@" in bet_text and any(x in bet_text for x in ["+", "-"]): return "Spread"
        return "Other"
    
    @staticmethod
    def calculate_metrics(parlays):
        metrics = defaultdict(lambda: {"win": 0, "loss": 0, "pending": 0})
        for parlay in parlays:
            for bet in parlay["bets"]:
                if bet["owner"] != "TBD":
                    result = bet["result"].lower()
                    if result in metrics[bet["owner"]]:
                        metrics[bet["owner"]][result] += 1
                    else:
                        st.warning(f"Unexpected result '{bet['result']}' for owner '{bet['owner']}'.")
        return metrics

class ParlayUI:
    def __init__(self):
        self.data = ParlayData()
        self.analyzer = BetAnalyzer()
        
    def setup_page(self):
        # Display Logo in Sidebar
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
            week = st.number_input("NFL Week", 1, 18, 1)
            year = st.number_input("Year", 2024, 2030, 2024)
        with col2:
            submitter = st.selectbox("Submitter", st.session_state.data["settings"]["owners"])
            date = st.date_input("Date", datetime.today())
        
        week_key = self.data.get_week_key(week, year)
        existing_bets = self.data.get_week_bets(week, year)
        submitted_owners = {bet["owner"] for parlay in existing_bets for bet in parlay["bets"] if bet["owner"] != "TBD"}
        
        available_owners = [o for o in st.session_state.data["settings"]["owners"] if o not in submitted_owners]
        
        st.markdown("""
        ### How to Format Your Bet:
        - **Team vs Team Over/Under X.Y**
          - *Example:* `Under 48.5 Cardinals vs Dolphins`
        - **Anytime TD Scorer - WR Team @ Team**
          - *Example:* `Anytime TD Scorer - WR Cardinals @ NE Patriots`
        - **Spread Bets with SGP (Same Game Parlay)**
          - *Example:* `+5 3Pks SGP TEN Titans @ DET Lions`
        - **Multiple Picks with SGP**
          - *Example:* `2 Picks SGP KC Chiefs @ LA Chargers`
        - **Any other formats should clearly state the type and details of the bet.**
        """)
        
        bet = st.text_input("Your Bet")
        owner = st.selectbox("Select Owner", ["TBD"] + available_owners)
        
        if st.button("Submit Bet"):
            if bet and owner != "TBD":
                if not existing_bets:
                    st.session_state.data["parlay_history"][week_key] = [{
                        "owner": submitter,
                        "date": date.strftime("%Y-%m-%d"),
                        "total_odds": "+0",
                        "bets": [],
                        "parlay_result": "PENDING"
                    }]
                
                new_bet = {
                    "bet": bet,
                    "result": "PENDING",
                    "owner": owner
                }
                
                st.session_state.data["parlay_history"][week_key][0]["bets"].append(new_bet)
                self.data.save()
                st.success(f"Bet submitted for {owner}!")
            else:
                st.error("Please enter a valid bet and select an owner.")
    
    def show_owner_stats(self):
        st.header("Owner Statistics")
        
        owner_stats = defaultdict(lambda: {"total_bets": 0, "win": 0, "loss": 0, "pending": 0})
        
        for parlays in st.session_state.data["parlay_history"].values():
            for parlay in parlays:
                for bet in parlay["bets"]:
                    if bet["owner"] != "TBD":
                        owner = bet["owner"]
                        result = bet["result"].lower()
                        owner_stats[owner]["total_bets"] += 1
                        if result in owner_stats[owner]:
                            owner_stats[owner][result] += 1
                        else:
                            st.warning(f"Unexpected result '{bet['result']}' for owner '{owner}'.")
        
        stats_df = pd.DataFrame([
            {
                "Owner": owner,
                "Total Bets": stats["total_bets"],
                "Wins": stats["win"],
                "Losses": stats["loss"],
                "Pending": stats["pending"],
                "Win Rate": f"{(stats['win']/(stats['win'] + stats['loss'])*100):.1f}%" if (stats['win'] + stats['loss']) > 0 else "0%"
            }
            for owner, stats in owner_stats.items()
        ])
        
        st.dataframe(stats_df)
        
        fig = px.bar(stats_df, 
                    x="Owner", 
                    y=["Wins", "Losses", "Pending"],
                    title="Owner Performance",
                    barmode="stack")
        st.plotly_chart(fig)

    def show_leaderboard(self):
        st.header("Leaderboard")
        
        metrics = self.analyzer.calculate_metrics(
            [parlay for parlays in st.session_state.data["parlay_history"].values() for parlay in parlays]
        )
        
        leaderboard = pd.DataFrame([
            {
                "Owner": owner,
                "Wins": stats["win"],
                "Losses": stats["loss"],
                "Win Rate (%)": (stats['win'] / (stats['win'] + stats['loss']) * 100) if (stats['win'] + stats['loss']) > 0 else 0,
                "Total Bets": stats["win"] + stats["loss"] + stats["pending"]
            }
            for owner, stats in metrics.items() if owner != "TBD"
        ])
        
        # Sort by 'Win Rate (%)' descending
        leaderboard = leaderboard.sort_values("Win Rate (%)", ascending=False)
        
        # Add formatted 'Win Rate' column
        leaderboard["Win Rate"] = leaderboard["Win Rate (%)"].apply(lambda x: f"{x:.1f}%")
        leaderboard = leaderboard.drop(columns=["Win Rate (%)"])
        
        st.dataframe(leaderboard)
    
    def manage_results(self):
        st.header("Manage Results")
        
        weeks = [
            (week_key, int(week_key.split(' ')[1]), int(week_key.split(' ')[2]))
            for week_key in st.session_state.data["parlay_history"].keys()
        ]
        
        if not weeks:
            st.warning("No parlays to manage")
            return
            
        selected_week_display = st.selectbox(
            "Select Week",
            [f"Week {week} {year}" for _, week, year in weeks]
        )
        
        selected_week_tuple = next(
            (wk for wk in weeks if f"Week {wk[1]} {wk[2]}" == selected_week_display),
            None
        )
        
        if selected_week_tuple:
            week_key = selected_week_tuple[0]
            parlay = st.session_state.data["parlay_history"][week_key][0]
            
            st.subheader(f"Managing Results for {week_key}")
            
            # Initialize session state for edit mode
            if 'edit_mode' not in st.session_state:
                st.session_state.edit_mode = {}
            
            for i, bet in enumerate(parlay["bets"]):
                with st.expander(f"Bet {i+1}: {bet['bet']} ({bet['owner']})"):
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        # Editable fields
                        new_bet_text = st.text_input(f"Bet {i+1} Text", value=bet["bet"], key=f"bet_text_{week_key}_{i}")
                        new_owner = st.selectbox(f"Owner for Bet {i+1}", st.session_state.data["settings"]["owners"], index=st.session_state.data["settings"]["owners"].index(bet["owner"]), key=f"bet_owner_{week_key}_{i}")
                    
                    with col2:
                        # Result selection
                        new_result = st.selectbox(
                            f"Result for Bet {i+1}",
                            ["PENDING", "WIN", "LOSS"],
                            index=["PENDING", "WIN", "LOSS"].index(bet["result"].upper()),
                            key=f"bet_result_{week_key}_{i}"
                        )
                    
                    with col3:
                        # Delete button
                        if st.button(f"Delete Bet {i+1}", key=f"delete_bet_{week_key}_{i}"):
                            st.session_state.data["parlay_history"][week_key][0]["bets"].pop(i)
                            self.data.save()
                            st.success(f"Bet {i+1} deleted.")
                            st.experimental_rerun()
                    
                    # Update the bet details
                    if st.button(f"Update Bet {i+1}", key=f"update_bet_{week_key}_{i}"):
                        if new_bet_text and new_owner:
                            st.session_state.data["parlay_history"][week_key][0]["bets"][i]["bet"] = new_bet_text
                            st.session_state.data["parlay_history"][week_key][0]["bets"][i]["owner"] = new_owner
                            st.session_state.data["parlay_history"][week_key][0]["bets"][i]["result"] = new_result
                            self.data.save()
                            st.success(f"Bet {i+1} updated.")
                        else:
                            st.error("Bet text and owner cannot be empty.")
            
            if st.button("Save All Results"):
                self.data.save()
                st.success("All results saved!")
        else:
            st.error("Selected week not found.")
    
    def show_statistics(self):
        st.header("Overall Statistics")
        
        total_bets = 0
        wins = 0
        losses = 0
        pending = 0
        
        for parlays in st.session_state.data["parlay_history"].values():
            for parlay in parlays:
                for bet in parlay["bets"]:
                    if bet["owner"] != "TBD":
                        total_bets += 1
                        if bet["result"].lower() == "win": wins += 1
                        elif bet["result"].lower() == "loss": losses += 1
                        else: pending += 1
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Bets", total_bets)
        with col2:
            win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
            st.metric("Win Rate", f"{win_rate:.1f}%")
        with col3:
            st.metric("Pending Bets", pending)
    
    def show_history(self):
        st.header("Parlay History")
        
        view = st.radio("View", ["By Week", "All Time"])
        
        if view == "By Week":
            weeks = sorted(st.session_state.data["parlay_history"].keys(), key=lambda x: (int(x.split(' ')[1]), int(x.split(' ')[2])), reverse=True)
            selected_week = st.selectbox("Select Week", weeks)
            if selected_week:
                week_num, year = self.parse_week_key(selected_week)
                parlays = self.data.get_week_bets(week_num, year)
        else:
            parlays = [p for ps in st.session_state.data["parlay_history"].values() for p in ps]
        
        records = []
        for parlay in parlays:
            week_key = self.get_week_of_parlay(parlay)
            for bet in parlay["bets"]:
                if bet["owner"] != "TBD":
                    records.append({
                        "Week": week_key,
                        "Owner": bet["owner"],
                        "Bet": bet["bet"],
                        "Result": bet["result"]
                    })
        
        if records:
            df = pd.DataFrame(records)
            st.dataframe(df)
        else:
            st.info("No bets to display.")
    
    def reset_history(self):
        st.header("Reset History")
        
        st.warning("This action will erase all submitted bets and cannot be undone.")
        if st.button("Reset All Bets"):
            st.session_state.data["parlay_history"] = {}
            self.data.save()
            st.success("All bets have been reset. History is now empty.")
    
    def parse_week_key(self, week_key):
        try:
            parts = week_key.split(' ')
            week = int(parts[1])
            year = int(parts[2])
            return week, year
        except (IndexError, ValueError):
            st.error("Invalid week format.")
            return None, None
    
    def get_week_of_parlay(self, parlay):
        # Find the week key for a given parlay
        for week_key, parlays in st.session_state.data["parlay_history"].items():
            if parlay in parlays:
                return week_key
        return "Unknown"
    
    def run(self):
        self.setup_page()

if __name__ == "__main__":
    st.set_page_config(
        page_title="LoEJ Perfect Parlay",
        page_icon="🎲",
        layout="wide"
    )
    app = ParlayUI()
    app.run()
