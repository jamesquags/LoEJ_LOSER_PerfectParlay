import streamlit as st
import yaml
from datetime import datetime
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from collections import defaultdict
import shutil

def load_history():
    if Path("history.yaml").exists():
        with open("history.yaml", "r") as file:
            return yaml.safe_load(file)
    # Initialize with default settings if history.yaml does not exist
    return {
        "parlay_history": {},
        "settings": {
            "owners": ["Owner1", "Owner2"],  # Default owners; update as needed
            "parlay_fee": 10  # Default parlay fee; update as needed
        }
    }

def save_history(data):
    backup_history()  # Create a backup before saving
    with open("history.yaml", "w") as file:
        yaml.dump(data, file, sort_keys=False)

def get_week_key(week, year):
    return f"week_{week}_{year}"

def backup_history():
    backup_dir = Path("backups")
    backup_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = backup_dir / f"history_backup_{timestamp}.yaml"
    if Path("history.yaml").exists():
        shutil.copy("history.yaml", backup_file)

class ParlayTracker:
    def __init__(self):
        self.data = load_history()
        self.setup_page()
        
    def setup_page(self):
        # Display Logo in Sidebar
        logo_path = "logo.png"  # Ensure you have a logo.png in your project directory
        if Path(logo_path).exists():
            st.sidebar.image(logo_path, use_column_width=True)
        
        st.title("League of Extraordinary Jabrones")
        st.subheader("Weekly Low Scorer Parlay Tracker")
        
        page = st.sidebar.selectbox(
            "Navigation",
            ["Submit Parlay", "View History", "Manage Results", "Statistics", "Owner Stats", "Leaderboard", "Advanced Analytics", "Import/Export"]
        )
        
        if page == "Submit Parlay":
            self.show_submission_form()
        elif page == "View History":
            self.show_history()
        elif page == "Manage Results":
            self.manage_results()
        elif page == "Statistics":
            self.show_statistics()
        elif page == "Owner Stats":
            self.show_owner_stats()
        elif page == "Leaderboard":
            self.show_leaderboard()
        elif page == "Advanced Analytics":
            self.show_advanced_analytics()
        elif page == "Import/Export":
            self.show_import_export()
    
    def show_submission_form(self):
        st.header("Submit Weekly Parlay")
        
        col1, col2 = st.columns(2)
        with col1:
            week = st.number_input("NFL Week", min_value=1, max_value=18, value=1)
            year = st.number_input("Year", min_value=2024, max_value=2030, value=2024)
        with col2:
            owner = st.selectbox("Owner", self.data["settings"].get("owners", ["Owner1", "Owner2"]))
            date = st.date_input("Date", value=datetime.today())
        
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
        
        week_keys = sorted(self.data["parlay_history"].keys(), key=lambda x: (int(x.split('_')[2]), int(x.split('_')[1])))
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
                    key=f"result_{week_key}_{i}",
                    index=["PENDING", "WIN", "LOSS"].index(bet["result"])
                )
                bet["result"] = new_result
                
            with col2:
                owners = ["TBD"] + self.data["settings"].get("owners", ["Owner1", "Owner2"])
                new_owner = st.selectbox(
                    f"Owner for Bet {i+1}",
                    owners,
                    key=f"owner_{week_key}_{i}",
                    index=owners.index(bet["owner"]) if bet["owner"] in owners else 0
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
            st.metric("Money Collected", f"${total_parlays * self.data['settings'].get('parlay_fee', 10)}")
            
        # Visualization of results over time
        st.subheader("Results Over Time")
        records = []
        for week_key, parlay in self.data["parlay_history"].items():
            records.append({
                "Week": week_key,
                "Result": parlay["parlay_result"],
                "Wins": sum(1 for bet in parlay["bets"] if bet["result"] == "WIN"),
                "Losses": sum(1 for bet in parlay["bets"] if bet["result"] == "LOSS")
            })
        results_df = pd.DataFrame(records)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=results_df["Week"], y=results_df["Wins"], name="Wins"))
        fig.add_trace(go.Bar(x=results_df["Week"], y=results_df["Losses"], name="Losses"))
        fig.update_layout(barmode='stack', title="Weekly Results Distribution", xaxis_title="Week", yaxis_title="Number of Bets")
        st.plotly_chart(fig)
        
        # Additional Visualization: Parlay Outcomes
        st.subheader("Parlay Outcomes")
        outcomes = results_df["Result"].value_counts().reset_index()
        outcomes.columns = ["Outcome", "Count"]
        
        fig_outcomes = px.pie(outcomes, names='Outcome', values='Count', title='Parlay Outcomes Distribution')
        st.plotly_chart(fig_outcomes)
    
    def show_owner_stats(self):
        st.header("Owner Statistics")
        
        if not self.data["parlay_history"]:
            st.warning("No data available for owner statistics")
            return
            
        owner_stats = {owner: {"wins": 0, "losses": 0, "pending": 0} 
                      for owner in self.data["settings"].get("owners", ["Owner1", "Owner2"])}
        
        for parlay in self.data["parlay_history"].values():
            for bet in parlay["bets"]:
                if bet["owner"] != "TBD":
                    owner = bet["owner"]
                    if owner not in owner_stats:
                        owner_stats[owner] = {"wins": 0, "losses": 0, "pending": 0}
                    if bet["result"] == "WIN":
                        owner_stats[owner]["wins"] += 1
                    elif bet["result"] == "LOSS":
                        owner_stats[owner]["losses"] += 1
                    else:
                        owner_stats[owner]["pending"] += 1
        
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
                    barmode="stack",
                    labels={"value": "Number of Bets", "variable": "Result"})
        st.plotly_chart(fig)
    
    def show_leaderboard(self):
        st.header("Leaderboard")
        
        # Calculate various metrics for each owner
        owner_metrics = defaultdict(lambda: {
            "total_bets": 0,
            "wins": 0,
            "losses": 0,
            "pending": 0,
            "profit": 0,
            "win_streaks": [],
            "current_streak": 0,
            "best_week": {"week": None, "wins": 0},
            "worst_week": {"week": None, "losses": 0},
            "favorite_bet_type": defaultdict(int)
        })
        
        for week_key, parlay in self.data["parlay_history"].items():
            week_stats = defaultdict(lambda: {"wins": 0, "losses": 0})
            
            for bet in parlay["bets"]:
                if bet["owner"] != "TBD":
                    owner = bet["owner"]
                    owner_metrics[owner]["total_bets"] += 1
                    
                    # Track bet types (simple classification)
                    bet_type = self.classify_bet_type(bet["bet"])
                    owner_metrics[owner]["favorite_bet_type"][bet_type] += 1
                    
                    if bet["result"] == "WIN":
                        owner_metrics[owner]["wins"] += 1
                        owner_metrics[owner]["current_streak"] += 1
                        week_stats[owner]["wins"] += 1
                    elif bet["result"] == "LOSS":
                        owner_metrics[owner]["losses"] += 1
                        if owner_metrics[owner]["current_streak"] > 0:
                            owner_metrics[owner]["win_streaks"].append(
                                owner_metrics[owner]["current_streak"]
                            )
                        owner_metrics[owner]["current_streak"] = 0
                        week_stats[owner]["losses"] += 1
                    else:
                        owner_metrics[owner]["pending"] += 1
            
            # Update best/worst weeks
            for owner, stats in week_stats.items():
                if (stats["wins"] > owner_metrics[owner]["best_week"]["wins"] or 
                    owner_metrics[owner]["best_week"]["week"] is None):
                    owner_metrics[owner]["best_week"] = {
                        "week": week_key,
                        "wins": stats["wins"]
                    }
                if (stats["losses"] > owner_metrics[owner]["worst_week"]["losses"] or 
                    owner_metrics[owner]["worst_week"]["week"] is None):
                    owner_metrics[owner]["worst_week"] = {
                        "week": week_key,
                        "losses": stats["losses"]
                    }
        
        # Create leaderboard DataFrame
        leaderboard_data = []
        for owner, metrics in owner_metrics.items():
            total_decided = metrics["wins"] + metrics["losses"]
            win_rate = (metrics["wins"] / total_decided * 100) if total_decided > 0 else 0
            best_streak = max(metrics["win_streaks"] + [metrics["current_streak"]], default=0)
            favorite_bet = max(metrics["favorite_bet_type"].items(), key=lambda x: x[1])[0] if metrics["favorite_bet_type"] else "N/A"
            
            best_week = metrics['best_week']
            worst_week = metrics['worst_week']
            best_week_str = f"{best_week['wins']} wins ({best_week['week'].split('_')[1]})" if best_week['week'] else "N/A"
            worst_week_str = f"{worst_week['losses']} losses ({worst_week['week'].split('_')[1]})" if worst_week['week'] else "N/A"
            
            leaderboard_data.append({
                "Owner": owner,
                "Win Rate": f"{win_rate:.1f}%",
                "Wins": metrics["wins"],
                "Losses": metrics["losses"],
                "Best Streak": best_streak,
                "Best Week": best_week_str,
                "Worst Week": worst_week_str,
                "Favorite Bet Type": favorite_bet
            })
        
        df_leaderboard = pd.DataFrame(leaderboard_data)
        df_leaderboard = df_leaderboard.sort_values("Win Rate", ascending=False)
        
        # Display leaderboard with tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["Main Leaderboard", "Streaks & Records", "Betting Patterns", "Weekly Performance"])
        
        with tab1:
            st.dataframe(df_leaderboard[["Owner", "Win Rate", "Wins", "Losses"]])
            
            # Visualization of win rates
            fig = go.Figure(data=[
                go.Bar(name="Win Rate", 
                      x=df_leaderboard["Owner"],
                      y=[float(x.strip('%')) for x in df_leaderboard["Win Rate"]])
            ])
            fig.update_layout(title="Owner Win Rates", xaxis_title="Owner", yaxis_title="Win Rate (%)")
            st.plotly_chart(fig)
        
        with tab2:
            streak_data = df_leaderboard[["Owner", "Best Streak", "Worst Week"]]
            st.dataframe(streak_data)
            
            # Best Streak visualization
            fig = go.Figure(data=[
                go.Bar(name="Best Streak",
                      x=streak_data["Owner"],
                      y=streak_data["Best Streak"])
            ])
            fig.update_layout(title="Best Win Streaks by Owner", xaxis_title="Owner", yaxis_title="Best Streak")
            st.plotly_chart(fig)
        
        with tab3:
            patterns_data = df_leaderboard[["Owner", "Favorite Bet Type"]]
            st.dataframe(patterns_data)
            
            # Betting patterns visualization
            bet_type_counts = defaultdict(lambda: defaultdict(int))
            for _, row in df_leaderboard.iterrows():
                owner = row["Owner"]
                favorite_bet = row["Favorite Bet Type"]
                bet_type_counts[favorite_bet][owner] += 1
            
            fig = go.Figure()
            for bet_type, counts in bet_type_counts.items():
                fig.add_trace(go.Bar(
                    name=bet_type,
                    x=list(counts.keys()),
                    y=list(counts.values())
                ))
            fig.update_layout(
                title="Betting Patterns by Owner",
                barmode='stack',
                xaxis_title="Owner",
                yaxis_title="Number of Favorite Bets"
            )
            st.plotly_chart(fig)
        
        with tab4:
            weekly_perf = []
            for week_key, parlay in self.data["parlay_history"].items():
                weekly_perf.append({
                    "Week": week_key,
                    "Parlay Result": parlay["parlay_result"]
                })
            df_weekly = pd.DataFrame(weekly_perf)
            df_weekly['Week_Number'] = df_weekly['Week'].apply(lambda x: int(x.split('_')[1]))
            df_weekly = df_weekly.sort_values("Week_Number")
            
            fig = px.line(df_weekly, x="Week_Number", y="Parlay Result", title="Weekly Parlay Results")
            st.plotly_chart(fig)
    
    def show_advanced_analytics(self):
        st.header("Advanced Analytics")
        
        if not self.data["parlay_history"]:
            st.warning("No data available for advanced analytics")
            return
        
        tab1, tab2, tab3 = st.tabs(["Trend Analysis", "Correlation Analysis", "Performance Insights"])
        
        with tab1:
            st.subheader("Win Rate Trends Over Time")
            
            # Prepare trend data
            trend_data = []
            for week_key, parlay in self.data["parlay_history"].items():
                week_num = int(week_key.split('_')[1])
                wins = sum(1 for bet in parlay["bets"] if bet["result"] == "WIN")
                losses = sum(1 for bet in parlay["bets"] if bet["result"] == "LOSS")
                win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
                
                trend_data.append({
                    "Week": week_num,
                    "Win Rate": win_rate,
                    "Total Bets": wins + losses
                })
            
            df_trends = pd.DataFrame(trend_data).sort_values("Week")
            
            # Rolling average visualization
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_trends["Week"],
                y=df_trends["Win Rate"],
                mode='lines+markers',
                name='Weekly Win Rate'
            ))
            fig.add_trace(go.Scatter(
                x=df_trends["Week"],
                y=df_trends["Win Rate"].rolling(window=3).mean(),
                mode='lines',
                name='3-Week Moving Average',
                line=dict(dash='dash')
            ))
            fig.update_layout(title="Win Rate Trends", xaxis_title="Week", yaxis_title="Win Rate (%)")
            st.plotly_chart(fig)
        
        with tab2:
            st.subheader("Correlation Analysis")
            
            # Analyze correlations between various metrics
            corr_data = []
            for week_key, parlay in self.data["parlay_history"].items():
                try:
                    odds = float(parlay["total_odds"].replace('+', '').replace('-', ''))
                except ValueError:
                    odds = 0
                wins = sum(1 for bet in parlay["bets"] if bet["result"] == "WIN")
                
                corr_data.append({
                    "Odds": odds,
                    "Wins": wins,
                    "Success": 1 if parlay["parlay_result"] == "WIN" else 0
                })
            
            df_corr = pd.DataFrame(corr_data)
            
            if not df_corr.empty and len(df_corr) > 1:
                # Visualization of odds vs. wins
                fig = px.scatter(df_corr, 
                               x="Odds", 
                               y="Wins",
                               color="Success",
                               title="Correlation between Odds and Wins",
                               labels={"Success": "Parlay Success"})
                st.plotly_chart(fig)
                
                # Calculate and display correlation coefficient
                correlation = df_corr["Odds"].corr(df_corr["Wins"])
                st.metric("Correlation between Odds and Wins", f"{correlation:.2f}")
            else:
                st.warning("Not enough data for correlation analysis.")
        
        with tab3:
            st.subheader("Performance Insights")
            
            # Analyze performance by bet type
            bet_type_performance = defaultdict(lambda: {"wins": 0, "losses": 0})
            for parlay in self.data["parlay_history"].values():
                for bet in parlay["bets"]:
                    if bet["result"] != "PENDING":
                        bet_type = self.classify_bet_type(bet["bet"])
                        if bet["result"] == "WIN":
                            bet_type_performance[bet_type]["wins"] += 1
                        else:
                            bet_type_performance[bet_type]["losses"] += 1
            
            # Calculate and display win rates by bet type
            performance_data = []
            for bet_type, stats in bet_type_performance.items():
                total = stats["wins"] + stats["losses"]
                win_rate = (stats["wins"] / total * 100) if total > 0 else 0
                performance_data.append({
                    "Bet Type": bet_type,
                    "Win Rate": win_rate,
                    "Total Bets": total
                })
            
            df_performance = pd.DataFrame(performance_data)
            
            # Visualization of performance by bet type
            fig = px.bar(df_performance,
                        x="Bet Type",
                        y="Win Rate",
                        color="Total Bets",
                        title="Performance by Bet Type",
                        labels={"Win Rate": "Win Rate (%)", "Total Bets": "Total Bets"})
            st.plotly_chart(fig)
    
    def classify_bet_type(self, bet_text):
        """Classify bet type based on bet text."""
        bet_text = bet_text.lower()
        if "td scorer" in bet_text or "touchdown" in bet_text:
            return "Touchdown Props"
        elif "under" in bet_text or "over" in bet_text:
            return "Totals"
        elif "sgp" in bet_text:
            return "Same Game Parlay"
        elif "passing" in bet_text:
            return "Passing Props"
        elif "@" in bet_text and any(x in bet_text for x in ["+", "-"]):
            return "Spread"
        else:
            return "Other"
    
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
                    "Year": week_key.split("_")[2],
                    "Owner": entry["owner"],
                    "Date": entry["date"],
                    "Total Odds": entry["total_odds"],
                    "Result": entry["parlay_result"],
                    "Wins": sum(1 for bet in entry["bets"] if bet["result"] == "WIN"),
                    "Losses": sum(1 for bet in entry["bets"] if bet["result"] == "LOSS")
                }
                records.append(record)
            df = pd.DataFrame(records)
            df = df.sort_values(["Year", "Week"], ascending=[False, False])
            
        else:
            records = []
            for week_key, entry in self.data["parlay_history"].items():
                for i, bet in enumerate(entry["bets"], 1):
                    record = {
                        "Week": week_key.split("_")[1],
                        "Year": week_key.split("_")[2],
                        "Bet Number": i,
                        "Bet": bet["bet"],
                        "Owner": bet["owner"],
                        "Result": bet["result"]
                    }
                    records.append(record)
            df = pd.DataFrame(records)
            df = df.sort_values(["Year", "Week", "Bet Number"], ascending=[False, False, True])
        
        st.dataframe(df)
        
        if st.button("Export to CSV"):
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name='parlay_history.csv',
                mime='text/csv'
            )
            st.success("Exported to parlay_history.csv")
    
    def show_import_export(self):
        st.header("Import / Export Data")
        
        # Export
        st.subheader("Export History")
        if st.button("Export history.yaml"):
            if Path("history.yaml").exists():
                with open("history.yaml", "r") as file:
                    data = file.read()
                st.download_button(
                    label="Download history.yaml",
                    data=data,
                    file_name="history.yaml",
                    mime="text/yaml"
                )
                st.success("history.yaml exported successfully!")
            else:
                st.error("history.yaml does not exist.")
        
        # Import
        st.subheader("Import History")
        uploaded_file = st.file_uploader("Upload history.yaml", type=["yaml"])
        if uploaded_file is not None:
            try:
                imported_data = yaml.safe_load(uploaded_file)
                if "parlay_history" in imported_data and "settings" in imported_data:
                    self.data = imported_data
                    save_history(self.data)
                    st.success("History imported successfully!")
                else:
                    st.error("Invalid history.yaml format.")
            except Exception as e:
                st.error(f"Error importing history.yaml: {e}")
    
if __name__ == "__main__":
    tracker = ParlayTracker()
