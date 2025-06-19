from flask import Flask, render_template, request, jsonify, session, url_for
import pandas as pd
import numpy as np
import os
import requests
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import logging
import random
from urllib.parse import quote
from werkzeug.utils import secure_filename
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    try:
        import sys
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
        from PIL import Image
        PIL_AVAILABLE = True
    except:
        pass

# -------------------- Initialize Flask App --------------------
app = Flask(__name__, static_url_path='/static', static_folder='cricket-team-prediction/static')
app.secret_key = '23c6a9eb7d6456ae6bf0472cbed52f2d'

# Configure paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_IMAGES_PATH = os.path.join(BASE_DIR, 'cricket-team-prediction', 'static', 'images')
PLAYER_IMAGES_PATH = os.path.join(STATIC_IMAGES_PATH, 'players')

# Ensure directories exist
os.makedirs(PLAYER_IMAGES_PATH, exist_ok=True)

# Configure Logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
app.logger.setLevel(logging.DEBUG)

# API Configuration (Optional - comment out if not using)
CRICAPI_KEY = "7fd62dff-20cc-47ca-b178-a97aabb9183d"  # Register at https://www.cricapi.com/
CRICAPI_BASE_URL = "https://cricapi.com/api"

# -------------------- Image Handling Functions --------------------
def get_player_image_from_api(player_name):
    """Fetch player image URL from CricAPI"""
    try:
        search_url = f"{CRICAPI_BASE_URL}/playerFinder?apikey={CRICAPI_KEY}&name={quote(player_name)}"
        response = requests.get(search_url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if data.get("data"):
            pid = data["data"][0]["pid"]
            image_url = f"{CRICAPI_BASE_URL}/playerStats?apikey={CRICAPI_KEY}&pid={pid}"
            img_response = requests.get(image_url, timeout=10)
            img_response.raise_for_status()
            return img_response.json().get("imageURL", "")
    except Exception as e:
        app.logger.error(f"API Error for {player_name}: {str(e)}")
    return ""

def resolve_player_photo(player_name, original_df):
    """Resolve player photo path with proper static URL generation"""
    try:
        clean_name = player_name.strip().lower()
        safe_name = secure_filename(clean_name.replace(' ', '_'))
        
        # 1. Check original dataset first
        if 'photo' in original_df.columns:
            photo = original_df.loc[original_df['playername'] == clean_name, 'photo'].values
            if len(photo) > 0 and photo[0] and str(photo[0]) != 'nan' and photo[0] != "default.jpg":
                return url_for('static', filename=f"images/{photo[0]}")
        
        # 2. Check local player images
        for ext in ['.jpg', '.jpeg', '.png']:
            img_path = os.path.join(PLAYER_IMAGES_PATH, f"{safe_name}{ext}")
            if os.path.exists(img_path):
                return url_for('static', filename=f"images/players/{safe_name}{ext}")
        
        # 3. Try API (comment out if not using API)
        if CRICAPI_KEY != "7fd62dff-20cc-47ca-b178-a97aabb9183d":
            api_url = get_player_image_from_api(player_name)
            if api_url:
                return api_url
        
        # 4. Final fallback
        return url_for('static', filename="images/default.jpg")
    
    except Exception as e:
        app.logger.error(f"Photo resolution error for {player_name}: {e}")
        return url_for('static', filename="images/default.jpg")

# -------------------- Load Data Function --------------------
def load_data(format_type):
    base_path = os.path.join(BASE_DIR, "datasets")
    file_paths = {
        "Batsmen": os.path.join(base_path, f"Batsman_data_{format_type}.xlsx"),
        "Bowlers": os.path.join(base_path, f"Bowlers_daata_{format_type}.xlsx"),
        "wicketkeeper": os.path.join(base_path, f"WicketKeepers_data_{format_type}.xlsx"),
        "All-Rounders": os.path.join(base_path, f"AllRounder_data_{format_type}.xlsx"),
        "match_details": os.path.join(base_path, f"MatchDetails_{format_type}.xlsx"),
        "playervsplayer": os.path.join(base_path, f"{format_type.lower()}.xlsx")
    }
    data, match_df, pvp_df = {}, None, None
    for role, path in file_paths.items():
        try:
            if os.path.exists(path):
                df = pd.read_excel(path)
                app.logger.debug(f"Loaded {role} data with {len(df)} rows: {df.columns.tolist()}")
                if not df.empty:
                    if role == "match_details":
                        match_df = df
                    elif role == "playervsplayer":
                        pvp_df = df
                    else:
                        data[role] = df
                else:
                    app.logger.warning(f"{role} data is empty: {path}")
            else:
                app.logger.warning(f"File not found: {path}")
        except Exception as e:
            app.logger.error(f"Error loading {role} data: {e}")

    if not data:
        app.logger.error(f"No player data loaded for format {format_type}")
    return data, match_df, pvp_df

# -------------------- Preprocess Data for ML --------------------
def preprocess_data(players_df):
    try:
        if players_df.empty:
            app.logger.warning("players_df is empty")
            return None, None, None
        feature_cols = players_df.select_dtypes(include=[np.number]).columns.drop('points', errors='ignore')
        if feature_cols.empty:
            app.logger.warning("No numeric feature columns found, using default points")
            return None, players_df.index, None
        X = players_df[feature_cols].fillna(0)
        y = players_df['points'].fillna(0) if 'points' in players_df.columns else None
        if y is None or y.isna().all():
            app.logger.warning("No valid points column, using default points")
            return None, players_df.index, None
        return X, y, None
    except Exception as e:
        app.logger.exception(f"Error in preprocess_data: {e}")
        return None, None, None

# -------------------- Train ML Models --------------------
def train_decision_tree(X, y):
    try:
        if X is None or y is None:
            app.logger.warning("Invalid data for DecisionTree")
            return None
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X, y > y.mean())
        return model
    except Exception as e:
        app.logger.exception(f"Error in train_decision_tree: {e}")
        return None

def train_xgboost(X, y):
    try:
        if X is None or y is None:
            app.logger.warning("Invalid data for XGBoost")
            return None
        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        model.fit(X, (y > y.mean()).astype(int))
        return model
    except Exception as e:
        app.logger.exception(f"Error in train_xgboost: {e}")
        return None

# -------------------- Predict Teams --------------------
def predict_teams(selected_players, original_players_df):
    try:
        app.logger.debug(f"Predicting with {len(selected_players)} players: {selected_players['PlayerName'].tolist()}")
        if selected_players.empty:
            app.logger.warning("No selected players provided")
            return pd.DataFrame()
        
        selected_players = selected_players.copy()
        selected_players['PlayerName'] = selected_players['PlayerName'].str.strip().str.lower()
        original_players_df.columns = original_players_df.columns.str.strip().str.lower()
        original_players_df['playername'] = original_players_df['playername'].str.strip().str.lower()

        if 'points' in original_players_df.columns and not original_players_df['points'].isna().all():
            selected_players['Points'] = selected_players['PlayerName'].map(
                original_players_df.set_index('playername')['points']
            ).fillna(0)
            selected_players['Role'] = selected_players['PlayerName'].map(
                original_players_df.set_index('playername')['role']
            ).fillna('Unknown')
        else:
            selected_players['Points'] = np.linspace(7.0, 9.0, len(selected_players))
            selected_players['Role'] = 'Unknown'

        selected_players['Photo'] = selected_players['PlayerName'].apply(
            lambda x: resolve_player_photo(x, original_players_df)
        )

        app.logger.debug(f"Final Assigned Data: {selected_players[['PlayerName', 'Role', 'Points', 'Photo']].to_dict(orient='records')}")
        return selected_players

    except Exception as e:
        app.logger.exception(f"Error in predict_teams: {e}")
        return pd.DataFrame()

# -------------------- Team Generation Functions --------------------
def generate_balanced_teams(sorted_players, num_teams=4, team_size=11):
    teams = []
    available_players = sorted_players.copy().sort_values(by='Points', ascending=False)
    
    if available_players.empty:
        app.logger.warning("No players provided to generate teams")
        return [pd.DataFrame() for _ in range(num_teams)]

    total_players = len(available_players)
    required_players = num_teams * team_size
    app.logger.debug(f"Total players: {total_players}, required: {required_players}")

    if total_players < required_players:
        app.logger.warning(f"Insufficient players: {total_players} provided, need {required_players}. Duplicating players.")
        duplicates_needed = (required_players - total_players) // total_players + 1
        available_players = pd.concat([available_players] * duplicates_needed, ignore_index=True)
        available_players = available_players.head(required_players)
        app.logger.debug(f"After duplication, total players: {len(available_players)}")

    full_pool = available_players.copy()

    # Team 1 (India)
    team1 = create_balanced_team(full_pool.sort_values(by='Points', ascending=False), team_size)
    if not team1.empty:
        teams.append(team1)
        app.logger.debug(f"Team 1 (India) created with {len(team1)} players: {team1['PlayerName'].tolist()}")
    else:
        teams.append(pd.DataFrame())

    remaining_players = full_pool[~full_pool.index.isin(teams[0].index)].copy()
    for i in range(1, num_teams):
        try:
            if len(remaining_players) >= team_size:
                shuffled_remaining = remaining_players.sample(frac=1, random_state=random.randint(1, 100)).reset_index(drop=True)
                team = create_balanced_team(shuffled_remaining, team_size)
                if not team.empty:
                    total_points = team['Points'].sum()
                    if total_points > 100:
                        team['Points'] = team['Points'] * (100 / total_points)
                    teams.append(team)
                else:
                    teams.append(pd.DataFrame())
            else:
                app.logger.warning(f"Not enough players for Team {i+1}: {len(remaining_players)} remaining")
                teams.append(pd.DataFrame())
        except Exception as e:
            app.logger.exception(f"Error generating team {i+1}: {e}")
            teams.append(pd.DataFrame())

    while len(teams) < num_teams:
        teams.append(pd.DataFrame())

    return teams

def create_balanced_team(players_df, team_size=11):
    try:
        if len(players_df) < team_size:
            app.logger.warning(f"Not enough players: {len(players_df)} available, need {team_size}")
            return pd.DataFrame()

        selected_players = []
        selected_names = set()
        role_counts = {'Batsmen': 0, 'Bowler': 0, 'wicketkeeper': 0, 'All-Rounder': 0}
        min_requirements = {'Batsmen': 4, 'Bowler': 3, 'wicketkeeper': 1, 'All-Rounder': 3}

        remaining_players = players_df.copy()

        for role, min_num in min_requirements.items():
            while role_counts[role] < min_num and len(selected_players) < team_size:
                candidates = remaining_players[
                    (remaining_players['Role'] == role) & 
                    (~remaining_players['PlayerName'].isin(selected_names))
                ]
                if candidates.empty:
                    app.logger.warning(f"No more {role} players available (need {min_num}, have {role_counts[role]})")
                    break

                top_player = candidates.sample(n=1, random_state=random.randint(1, 100)).iloc[0]
                selected_players.append(top_player.to_dict())
                selected_names.add(top_player['PlayerName'])
                role_counts[role] += 1
                remaining_players = remaining_players[remaining_players.index != top_player.name].copy()

        remaining_slots = team_size - len(selected_players)
        if remaining_slots > 0:
            remaining_players = remaining_players[~remaining_players['PlayerName'].isin(selected_names)]
            if not remaining_players.empty:
                extra_players = remaining_players.sample(n=remaining_slots, random_state=random.randint(1, 100))
                selected_players.extend(extra_players.to_dict('records'))

        team_df = pd.DataFrame(selected_players)[["PlayerName", "Role", "Team", "Points", "Photo"]]
        app.logger.debug(f"Team created: {team_df.to_dict(orient='records')}")
        return team_df
    except Exception as e:
        app.logger.exception(f"Error in create_balanced_team: {e}")
        return pd.DataFrame()

# -------------------- Flask Routes --------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route('/register')
def register():
    return render_template("register.html")

@app.route('/dashboard')
def dashboard():
    return render_template("dashboard.html")

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/logout")
def logout():
    return render_template("login.html")

@app.route("/profile")
def profile():
    return render_template("profile.html")

@app.route("/team")
def team():
    teams_data = session.get('formatted_teams', [])
    team1_name = session.get('team1_name', "Team 1")
    team2_name = session.get('team2_name', "Team 2")
    team3_name = "Team 3"
    team4_name = "Team 4"

    teams = [team_data for team_data in teams_data if team_data]
    app.logger.debug(f"Rendering {len(teams)} teams: {[len(t) for t in teams]}")
    return render_template("teamspredicted.html",
                         teams=teams,
                         team1_name=team1_name,
                         team2_name=team2_name,
                         team3_name=team3_name,
                         team4_name=team4_name)

@app.route('/predixi')
def predixi():
    return render_template("predixi.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    app.logger.debug("Entering /predict route")
    try:
        if request.method == "GET":
            return render_template("predixi.html")

        data = request.get_json()
        app.logger.debug(f"Received data: {data}")
        if not data or not data.get("players"):
            app.logger.error("No players provided")
            return jsonify({"error": "No players selected"}), 400

        format_type = data.get("format_type", "ODI")
        players = data.get("players")

        if len(players) != 22:
            app.logger.error(f"Expected 22 players, got {len(players)}")
            return jsonify({"error": "Please provide exactly 22 players (11 per team)"}), 400

        players_data, match_df, pvp_df = load_data(format_type)
        if not players_data:
            app.logger.error("No player data loaded")
            return jsonify({"error": "No player data available"}), 500

        players_df = pd.concat(players_data.values(), ignore_index=True)
        selected_df = players_df[players_df["PlayerName"].isin(players)]
        if len(selected_df) < 22:
            app.logger.warning(f"Only {len(selected_df)} of 22 players found in data")
            return jsonify({"error": "Some players not found in dataset"}), 400

        sorted_players = predict_teams(selected_df, players_df)
        if sorted_players.empty:
            sorted_players = selected_df.copy()
            sorted_players['Points'] = np.linspace(7.0, 9.0, len(selected_df))
            sorted_players['Photo'] = 'images/default.jpg'

        teams = generate_balanced_teams(sorted_players, num_teams=4, team_size=11)
        formatted_teams = [team.to_dict(orient='records') if not team.empty else [] for team in teams]
        session['formatted_teams'] = formatted_teams
        session['team1_name'] = data.get("team1_name", "Team 1")
        session['team2_name'] = data.get("team2_name", "Team 2")

        return jsonify({"status": "success"})

    except Exception as e:
        app.logger.exception(f"Error in /predict: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# -------------------- Run Flask App --------------------
if __name__ == "__main__":
    # Create default image if missing
    default_img = os.path.join(STATIC_IMAGES_PATH, 'default.jpg')
    if not os.path.exists(default_img):
        try:
            img = Image.new('RGB', (150, 150), color='gray')
            img.save(default_img)
            app.logger.info(f"Created default image at {default_img}")
        except Exception as e:
            app.logger.error(f"Could not create default image: {e}")

    app.logger.info(f"Static images path: {STATIC_IMAGES_PATH}")
    app.logger.info(f"Player images path: {PLAYER_IMAGES_PATH}")
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
