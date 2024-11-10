import streamlit as st
import yaml
from datetime import datetime
import pandas as pd
from pathlib import Path
import plotly.express as px
from collections import defaultdict
import shutil

class ParlayData:
    def __init__(self):
        self.data = self.load_history()
        
    @staticmethod
    def load_history():
        if Path("history.yaml").exists():
            with open("history.yaml", "r") as file:
                data = yaml.safe_load(file)
                return data if {"parlay_history", "settings"} <= set(data.keys()) else ParlayData.get_default_data()
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
        backup_dir = Path("backups")
        backup_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if Path("history.yaml").exists():
            shutil.copy("history.yaml", backup_dir / f"history_backup_{timestamp}.yaml")
        with open("history.yaml", "w") as file:
            yaml.dump(self.data, file, sort_keys=False)
    
    def get_week_key(self, week, year):
        return f"Week {week} {year}"
    
    def get_week_bets(self, week, year):
        week_key = self.get_week_key(week, year)
        return self.data["parlay_history"].get(week_key, [])
    
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
        st.title("League of Extraordinary Jabrones")
        st.subheader("Weekly Low Scorer Parlay Tracker")
        
        pages = {
            "Submit Parlay": self.show_submission_form,
            "View History": self.show_history,
            "Manage Results": self.manage_results,
            "Statistics": self.show_statistics,
            "Owner Stats": self.show_owner_stats,
            "Leaderboard": self.show_leaderboard,
            "Reset History": self.reset_history  # Added Reset History page
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
            submitter = st.selectbox("Submitter", self.data.data["settings"]["owners"])
            date = st.date_input("Date", datetime.today())
        
        # Get already submitted bets for this week
        week_key = self.data.get_week_key(week, year)
        existing_bets = self.data.get_week_bets(week, year)
        submitted_owners = {bet["owner"] for parlay in existing_bets for bet in parlay["bets"] if bet["owner"] != "TBD"}
        
        # Show available owners
        available_owners = [o for o in self.data.data["settings"]["owners"] if o not in submitted_owners]
        
        # Removed the Total Odds input
        # st.text_input("Total Odds")  # Removed
        
        # Add detailed syntax instructions
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
                    # Create new week entry
                    self.data.data["parlay_history"][week_key] = [{
                        "owner": submitter,
                        "date": date.strftime("%Y-%m-%d"),
                        "total_odds": "+0",  # Placeholder since odds are tracked internally
                        "bets": [],
                        "parlay_result": "PENDING"
                    }]
                
                # Add bet to existing parlay
                new_bet = {
                    "bet": bet,
                    "result": "PENDING",
                    "owner": owner
                }
                
                self.data.data["parlay_history"][week_key][0]["bets"].append(new_bet)
                self.data.save()
                st.success(f"Bet submitted for {owner}!")
            else:
                st.error("Please enter a valid bet and select an owner.")

    def show_owner_stats(self):
        st.header("Owner Statistics")
        
        # Calculate owner stats
        owner_stats = defaultdict(lambda: {"total_bets": 0, "win": 0, "loss": 0, "pending": 0})
        
        for parlays in self.data.data["parlay_history"].values():
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
        
        # Create DataFrame for display
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
        
        # Visualization
        fig = px.bar(stats_df, 
                    x="Owner", 
                    y=["Wins", "Losses", "Pending"],
                    title="Owner Performance",
                    barmode="stack")
        st.plotly_chart(fig)

    def show_leaderboard(self):
        st.header("Leaderboard")
        
        metrics = self.analyzer.calculate_metrics(
            [parlay for parlays in self.data.data["parlay_history"].values() for parlay in parlays]
        )
        
        leaderboard = pd.DataFrame([
            {
                "Owner": owner,
                "Wins": stats["win"],
                "Losses": stats["loss"],
                "Win Rate": f"{(stats['win']/(stats['win'] + stats['loss'])*100):.1f}%" if (stats['win'] + stats['loss']) > 0 else "0%",
                "Total Bets": stats["win"] + stats["loss"] + stats["pending"]
            }
            for owner, stats in metrics.items() if owner != "TBD"
        ]).sort_values("Win Rate", ascending=False)
        
        st.dataframe(leaderboard)

    def manage_results(self):
        st.header("Manage Results")
        
        weeks = [
            (week_key, int(week_key.split(' ')[1]), int(week_key.split(' ')[2]))
            for week_key in self.data.data["parlay_history"].keys()
        ]
        
        if not weeks:
            st.warning("No parlays to manage")
            return
            
        selected_week_display = st.selectbox(
            "Select Week",
            [f"Week {week} {year}" for _, week, year in weeks]
        )
        
        # Find the corresponding week_key
        selected_week_tuple = next(
            (wk for wk in weeks if f"Week {wk[1]} {wk[2]}" == selected_week_display),
            None
        )
        
        if selected_week_tuple:
            week_key = selected_week_tuple[0]
            parlay = self.data.data["parlay_history"][week_key][0]
            
            for i, bet in enumerate(parlay["bets"]):
                st.markdown(f"**Bet {i+1}**: {bet['bet']} ({bet['owner']})")
                parlay["bets"][i]["result"] = st.selectbox(
                    f"Result for {bet['owner']}'s bet",
                    ["PENDING", "WIN", "LOSS"],
                    index=["PENDING", "WIN", "LOSS"].index(bet["result"].upper())
                )
            
            if st.button("Save Results"):
                self.data.save()
                st.success("Results updated!")
        else:
            st.error("Selected week not found.")

    def show_statistics(self):
        st.header("Overall Statistics")
        
        total_bets = 0
        wins = 0
        losses = 0
        pending = 0
        
        for parlays in self.data.data["parlay_history"].values():
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
            st.metric("Win Rate", f"{(wins/(wins+losses)*100):.1f}%" if wins+losses > 0 else "0%")
        with col3:
            st.metric("Pending Bets", pending)

    def show_history(self):
        st.header("Parlay History")
        
        view = st.radio("View", ["By Week", "All Time"])
        
        if view == "By Week":
            weeks = sorted(self.data.data["parlay_history"].keys())
            selected_week = st.selectbox("Select Week", weeks)
            parlays = self.data.get_week_bets(*self.parse_week_key(selected_week))
        else:
            parlays = [p for ps in self.data.data["parlay_history"].values() for p in ps]
        
        records = []
        for parlay in parlays:
            for bet in parlay["bets"]:
                if bet["owner"] != "TBD":
                    records.append({
                        "Week": parlay.get("week", ""),
                        "Owner": bet["owner"],
                        "Bet": bet["bet"],
                        "Result": bet["result"]
                    })
        
        df = pd.DataFrame(records)
        st.dataframe(df)

    def reset_history(self):
        st.header("Reset History")
        
        st.warning("This action will erase all submitted bets and cannot be undone.")
        if st.button("Reset All Bets"):
            self.data.data["parlay_history"] = {}
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

    def run(self):
        self.setup_page()

if __name__ == "__main__":
    ParlayUI().run()
