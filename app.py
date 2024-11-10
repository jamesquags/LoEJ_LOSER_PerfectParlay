import streamlit as st
import yaml
from datetime import datetime
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

def load_history():
    if Path("history.yaml").exists():
        with open("history.yaml", "r") as file:
            return yaml.safe_load(file)
    return {"parlay_history": {}, "settings": {}}

def save_history(data):
    with open("history.yaml", "w") as file:
        yaml.dump(data, file, sort_keys=False)

def get_week_key(week, year):
    return f"week_{week}_{year}"

class ParlayTracker:
    def __init__(self):
        self.data = load_history()
        self.setup_page()
        
    def setup_page(self):
        st.title("League of Extraordinary Jabrones")
        st.subheader("Weekly Low Scorer Parlay Tracker")
        
        page = st.sidebar.selectbox(
            "Navigation",
            ["Submit Parlay", "View History", "Manage Results", "Statistics", "Owner Stats"]
        )
        
        if page == "Submit Parlay":
            self.show_submission_form()
        elif page == "View History":
            self.show_history()
        elif page == "Manage Results":
            self.manage_results()
        elif page == "Statistics":
            self.show_statistics()
        else:
            self.show_owner_stats()

    def show_submission_form(self):
        st.header("Submit Weekly Parlay")
        
        col1, col2 = st.columns(2)
        with col1:
            week = st.number_input("NFL Week", min_value=1, max_value=18)
            year = st.number_input("Year", min_value=2024, max_value=2030, value=2024)
        with col2:
            owner = st.selectbox("Owner", self.data["settings"]["owners"])
            date = st.date_input("Date")
        
        total_odds = st.text_input("Total Odds (e.g., +7703)")
        
        st.subheader("Enter 10 Bets")
        bets = []
        for i in range(10):
            bet = st.text_input(f"Bet {i+1}")
            if bet:
                bets.append({
                    "bet": bet,
                    "result": "PENDING",
                    "owner": "TBD"
                })
            
        if st.button("Submit Parlay"):
            if len(bets) == 10:
                week_key = get_week_key(week, year)
                
                new_entry = {
                    "owner": owner,
                    "date": date.strftime("%Y-%m-%d"),
                    "total_odds": total_odds,
                    "bets": bets,
                    "parlay_result": "PENDING"
                }
                
                self.data["parlay_history"][week_key] = new_entry
                save_history(self.data)
                st.success("Parlay submitted successfully!")
            else:
                st.error("Please enter all 10 bets")

    def manage_results(self):
        st.header("Manage Results")
        
        week_keys = list(self.data["parlay_history"].keys())
        if not week_keys:
            st.warning("No parlays to update")
            return
            
        week_key = st.selectbox("Select Week", week_keys)
        parlay = self.data["parlay_history"][week_key]
        
        st.subheader("Update Individual Bets")
        
        for i, bet in enumerate(parlay["bets"]):
            st.markdown(f"**Bet {i+1}**: {bet['bet']}")
            col1, col2 = st.columns(2)
            
            with col1:
                new_result = st.selectbox(
                    f"Result for Bet {i+1}",
                    ["PENDING", "WIN", "LOSS"],
                    key=f"result_{i}",
                    index=["PENDING", "WIN", "LOSS"].index(bet["result"])
                )
                bet["result"] = new_result
                
            with col2:
                new_owner = st.selectbox(
                    f"Owner for Bet {i+1}",
                    ["TBD"] + self.data["settings"]["owners"],
                    key=f"owner_{i}",
                    index=(["TBD"] + self.data["settings"]["owners"]).index(bet["owner"])
                )
                bet["owner"] = new_owner
                
        st.subheader("Update Parlay Result")
        parlay_result = st.selectbox(
            "Overall Parlay Result",
            ["PENDING", "WIN", "LOSS"],
            index=["PENDING", "WIN", "LOSS"].index(parlay["parlay_result"])
        )
        
        if st.button("Save Updates"):
            parlay["parlay_result"] = parlay_result
            save_history(self.data)
            st.success("Results updated successfully!")

    def show_statistics(self):
        st.header("Parlay Statistics")
        
        if not self.data["parlay_history"]:
            st.warning("No data available for statistics")
            return
            
        # Overall Statistics
        st.subheader("Overall Statistics")
        total_parlays = len(self.data["parlay_history"])
        total_bets = total_parlays * 10
        
        # Calculate win rates
        parlay_wins = sum(1 for p in self.data["parlay_history"].values() 
                         if p["parlay_result"] == "WIN")
        
        bet_results = []
        for parlay in self.data["parlay_history"].values():
            for bet in parlay["bets"]:
                if bet["result"] != "PENDING":
                    bet_results.append(bet["result"])
        
        bet_wins = sum(1 for r in bet_results if r == "WIN")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Parlays", total_parlays)
            st.metric("Parlay Win Rate", f"{(parlay_wins/total_parlays)*100:.1f}%" if total_parlays > 0 else "0%")
        with col2:
            st.metric("Total Bets", total_bets)
            st.metric("Individual Bet Win Rate", f"{(bet_wins/len(bet_results))*100:.1f}%" if bet_results else "0%")
        with col3:
            st.metric("Money Collected", f"${total_parlays * self.data['settings']['parlay_fee']}")
            
        # Visualization of results over time
        st.subheader("Results Over Time")
        results_df = pd.DataFrame([
            {
                "Week": week_key,
                "Result": parlay["parlay_result"],
                "Wins": sum(1 for bet in parlay["bets"] if bet["result"] == "WIN"),
                "Losses": sum(1 for bet in parlay["bets"] if bet["result"] == "LOSS")
            }
            for week_key, parlay in self.data["parlay_history"].items()
        ])
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=results_df["Week"], y=results_df["Wins"], name="Wins"))
        fig.add_trace(go.Bar(x=results_df["Week"], y=results_df["Losses"], name="Losses"))
        fig.update_layout(barmode='stack', title="Weekly Results Distribution")
        st.plotly_chart(fig)

    def show_owner_stats(self):
        st.header("Owner Statistics")
        
        if not self.data["parlay_history"]:
            st.warning("No data available for owner statistics")
            return
            
        owner_stats = {owner: {"wins": 0, "losses": 0, "pending": 0} 
                      for owner in self.data["settings"]["owners"]}
        
        for parlay in self.data["parlay_history"].values():
            for bet in parlay["bets"]:
                if bet["owner"] != "TBD":
                    owner_stats[bet["owner"]][bet["result"].lower()] += 1
        
        # Convert to DataFrame for display
        stats_df = pd.DataFrame([
            {
                "Owner": owner,
                "Wins": stats["wins"],
                "Losses": stats["losses"],
                "Pending": stats["pending"],
                "Win Rate": f"{(stats['wins']/(stats['wins'] + stats['losses'])*100):.1f}%" if (stats['wins'] + stats['losses']) > 0 else "0%"
            }
            for owner, stats in owner_stats.items()
        ])
        
        st.dataframe(stats_df)
        
        # Visualization
        fig = px.bar(stats_df, 
                    x="Owner", 
                    y=["Wins", "Losses", "Pending"],
                    title="Owner Performance Breakdown",
                    barmode="stack")
        st.plotly_chart(fig)

    def show_history(self):
        st.header("Parlay History")
        
        if not self.data["parlay_history"]:
            st.warning("No parlay history available")
            return
            
        view_type = st.radio("View Type", ["Summary", "Detailed"])
        
        if view_type == "Summary":
            records = []
            for week_key, entry in self.data["parlay_history"].items():
                record = {
                    "Week": week_key.split("_")[1],
                    "Owner": entry["owner"],
                    "Date": entry["date"],
                    "Total Odds": entry["total_odds"],
                    "Result": entry["parlay_result"],
                    "Wins": sum(1 for bet in entry["bets"] if bet["result"] == "WIN"),
                    "Losses": sum(1 for bet in entry["bets"] if bet["result"] == "LOSS")
                }
                records.append(record)
            df = pd.DataFrame(records)
            
        else:
            records = []
            for week_key, entry in self.data["parlay_history"].items():
                for i, bet in enumerate(entry["bets"], 1):
                    record = {
                        "Week": week_key.split("_")[1],
                        "Bet Number": i,
                        "Bet": bet["bet"],
                        "Owner": bet["owner"],
                        "Result": bet["result"]
                    }
                    records.append(record)
            df = pd.DataFrame(records)
        
        st.dataframe(df)
        
        if st.button("Export to CSV"):
            df.to_csv("parlay_history.csv", index=False)
            st.success("Exported to parlay_history.csv")

if __name__ == "__main__":
    tracker = ParlayTracker()
