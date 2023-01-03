import h5py
import bs4
import pandas as pd
import numpy as np
import requests
import os
from dateutil.parser import parse
import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


## HDF5 INITIALIZATION ##
date_now = datetime.datetime.now();
LAST_SEASON = date_now.year+1 if date_now.month >= 11 else date_now.year;
FIRST_SEASON = 2010;
NUMBER_OF_SEASON = LAST_SEASON-FIRST_SEASON;
RANGE = range(FIRST_SEASON,LAST_SEASON+1);

nba_hdf5_file = os.path.dirname(__file__)+"/static/nba_teams_file.hdf5";

NBA_TEAMS = {
    "ATL": (RANGE, None, "Atlanta Hawks"),
    "BOS": (RANGE, None, "Boston Celtics"),
    "BRK": (range(2013,LAST_SEASON+1), None, "Brooklyn Nets"),
    "NJN": (range(FIRST_SEASON,2012+1), "BRK"),
    "CHO": (range(2015,LAST_SEASON+1), None, "Charlotte Hornets"),
    "CHA": (range(FIRST_SEASON, 2014+1), "CHO"),
    "CHI": (RANGE, None, "Chicago Bulls"),
    "CLE": (RANGE, None, "Cleveland Cavaliers"),
    "DAL": (RANGE, None, "Dallas Mavericks"),
    "DEN": (RANGE, None, "Denver Nuggets"),
    "DET": (RANGE, None, "Detroit Pistons"),
    "GSW": (RANGE, None, "Golden State Warriors"),
    "HOU":(RANGE, None, "Houston Rockets"),
    "IND": (RANGE, None, "Indiana Pacers"),
    "LAC": (RANGE, None, "Los Angeles Clippers"),
    "LAL": (RANGE, None, "Los Angeles Lakers"),
    "MEM": (RANGE, None, "Memphis Grizzlies"),
    "MIA": (RANGE, None, "Miami Heat"),
    "MIL": (RANGE, None, "Milwaukee Bucks"),
    "MIN": (RANGE, None, "Minnesota Timberwolves"),
    "NOP": (range(2014,LAST_SEASON+1), None, "New Orleans Pelicans"),
    "NOH": (range(FIRST_SEASON,2013+1), "NOP"),
    "NYK": (RANGE, None, "New York Knicks"),
    "OKC": (RANGE, None, "Oklahoma City Thunder"),  
    "ORL": (RANGE, None, "Orlando Magic"),
    "PHI": (RANGE, None, "Philadelphia 76ers"),
    "PHO": (RANGE, None, "Phoenix Suns"),
    "POR": (RANGE, None, "Portland Trail Blazers"),
    "SAC": (RANGE, None, "Sacramento Kings"),
    "SAS": (RANGE, None, "San Antonio Spurs"),
    "TOR": (RANGE, None, "Toronto Raptors"),
    "UTA": (RANGE, None, "Utah Jazz"),
    "WAS": (RANGE, None, "Washington Wizards")
};

COLUMNS = 'Rk,G,Date,@,OppN,W/L,Tm,OppP,FG,FGA,FG%,3P,3PA,3P%,FT,FTA,FT%,ORB,TRB,AST,STL,BLK,TOV,PF,O_FG,O_FGA,O_FG%,O_3P,O_3PA,O_3P%,O_FT,O_FTA,O_FT%,O_ORB,O_TRB,O_AST,O_STL,O_BLK,O_TOV,O_PF';
COLUMNS = COLUMNS.split(',');

def initHDF5FileStructure():
    nba_teams_file = h5py.File(nba_hdf5_file, 'w');
    string_dtype = h5py.string_dtype(encoding='utf-8');

    for season in range(FIRST_SEASON,LAST_SEASON+1):
        nba_teams_file.create_group(f'{season}');

    for season in range(FIRST_SEASON,LAST_SEASON+1):
        for team in NBA_TEAMS:
            if NBA_TEAMS[team][1] is None:
                nba_teams_file[str(season)].create_dataset(team, (83,40), dtype=string_dtype);

    nba_teams_file.close();

# Pull all data from BasketBall-Reference's site.
def pullAllDataFromBR():
    nba_teams_file = h5py.File(nba_hdf5_file, 'r+');

    for team in NBA_TEAMS:
        print("TEAM ", team)
        season_range = NBA_TEAMS[team][0];
        new_acronym = NBA_TEAMS[team][1];

        for season in season_range:
            url = f"https://www.basketball-reference.com/teams/{team}/{str(season)}/gamelog";
            r = requests.get(url);
            soup_obj = bs4.BeautifulSoup(r.text, features="html.parser");

            for match_id in range(1,83):
                row = soup_obj.find(id="div_tgl_basic").tbody.find(id=f"tgl_basic.{str(match_id)}");

                if row is None:
                    break;

                row = row.get_text(",");
                row = row.split(',');

                if row[3] != '@':
                    row.insert(3, ' ');

                if new_acronym is None and match_id == 1:
                    nba_teams_file[str(season)][team][0] = COLUMNS;
                elif new_acronym is not None and match_id == 1:
                    nba_teams_file[str(season)][new_acronym][0] = COLUMNS;

                if new_acronym is None:
                    nba_teams_file[str(season)][team][match_id] = row;
                else:
                    nba_teams_file[str(season)][new_acronym][match_id] = row;

    nba_teams_file.close();


## PULL AND FILTER DATA ##

# Pull new data (from LAST_SEASON) from BasketBall-Reference's site.
def pullDataFromBRUpdate():
    nba_teams_file = h5py.File(nba_hdf5_file, 'r+');

    for team in NBA_TEAMS:
        new_acronym = NBA_TEAMS[team][1];
        season = LAST_SEASON;

        if new_acronym is not None:
            continue;

        url = f"https://www.basketball-reference.com/teams/{team}/{str(season)}/gamelog";
        r = requests.get(url);
        soup_obj = bs4.BeautifulSoup(r.text, features="html.parser");

        for match_id in range(1,83):
            row = soup_obj.find(id="div_tgl_basic").tbody.find(id=f"tgl_basic.{str(match_id)}");

            if row is None:
                break;

            row = row.get_text(",");
            row = row.split(',');

            if row[3] != '@':
                row.insert(3, ' ');

            if match_id == 1:
                nba_teams_file[str(season)][team][0] = COLUMNS;

            nba_teams_file[str(season)][team][match_id] = row;

    nba_teams_file.close();

# Prevents the <pullDataFromBRUpdate> function from pulling data on the BasketBall-Reference's site unnecessarily 
# when the NBA season is paused. End: first of May, Start: first of November.
def inhibitorPullDataFromBRUpdate(update_nba_data_job):
    if 5 <= datetime.datetime.now().month <= 10:
        update_nba_data_job.pause();
    else:
        update_nba_data_job.resume();

# pull pandas DataFrame containing informations about the <team> on the specified <season>
def pullDF(season, team):
    nba_teams_file = h5py.File(nba_hdf5_file, 'r+')

    DF = pd.DataFrame(nba_teams_file[season][team]);
    
    nba_teams_file.close();
    
    columns = DF.iloc[0].str.decode('utf-8');
    DF = DF[1:];
    DF.columns = columns;

    for col in DF.columns:
        DF[col] = pd.to_numeric(DF[col], errors='ignore');

    DF['Date'] = DF['Date'].str.decode('utf-8');
    DF['@'] = DF['@'].str.decode('utf-8');
    DF['OppN'] = DF['OppN'].str.decode('utf-8');
    DF['W/L'] = DF['W/L'].str.decode('utf-8');

    return DF.dropna();

# pull all matchs of all teams from year <start_year> to LAST_SEASON
# <start_year> allowed range: [FIRST_SEASON,LAST_SEASON]
def pullAllData(start_year):
    if start_year not in range(FIRST_SEASON,LAST_SEASON+1):
        raise ValueError('start_year value error');

    DF = None;

    for season in range(start_year, LAST_SEASON+1):
        for team in NBA_TEAMS:
            if NBA_TEAMS[team][1] is None:
                if DF is None:
                    DF = pullDF(str(season), team);
                else:
                    DF = pd.concat([pullDF(str(season), team), DF]);
            else:
                continue;

    return DF;


# pull all matchs of <team> from year <start_year> to LAST_SEASON
# <start_year> allowed range: [FIRST_SEASON,LAST_SEASON]
def pullDataTeam(start_year, team):
    if start_year not in range(FIRST_SEASON,LAST_SEASON+1):
        raise ValueError('start_year value error');

    DF = None;

    for season in range(start_year, LAST_SEASON+1):
        if DF is None:
            DF = pullDF(str(season), team);
        else:
            DF = pd.concat([pullDF(str(season), team), DF]);

    return DF;


# pull all matchs of <team> from year <start_year> to <end_year>
# <start_year> and <end_yeard> allowed range: [FIRST_SEASON,LAST_SEASON]
# <start_year> <= <end_yeard>
def pullDataTeamSeasonRange(start_year, end_year, team):
    if start_year not in range(FIRST_SEASON,LAST_SEASON+1):
        raise ValueError('start_year value error');

    DF = None;

    for season in range(start_year, end_year+1):
        if DF is None:
            DF = pullDF(str(season), team);
        else:
            DF = pd.concat([pullDF(str(season), team), DF]);

    return DF;


# pull all matchs of <team> from year <start_year> to LAST_SEASON
# <start_year> allowed range: [FIRST_SEASON,LAST_SEASON]
# Filter the team's games according to whether they are played at home or away.
def pullDataTeamAtHome(pull_Y, team, is_team_home):
    DF = pullDataTeam(pull_Y, team);

    if is_team_home:
        DF = DF[DF['@'] != '@'];
    else:
        DF = DF[DF['@'] == '@'];

    return DF;


# pull all matchs of <team> vs <opponent> from year <start_year> to LAST_SEASON
# <start_year> allowed range: [FIRST_SEASON,LAST_SEASON]
def pullDataTeamVSOpponent(pull_Y, team, opponent):
    DF = pullDataTeam(pull_Y, team);

    return DF[DF['OppN'] == opponent];


# pull all matchs of <team> vs <opponent> from year <start_year> to LAST_SEASON
# <start_year> allowed range: [FIRST_SEASON,LAST_SEASON]
# Filter the team's games according to whether they are played at home (team's home) or away (opponent's home).
def LD(pull_Y, team, opponent, is_team_home):
    DF = pullDataTeamVSOpponent(pull_Y, team, opponent);

    if is_team_home:
        DF = DF[DF['@'] != '@'];
    else:
        DF = DF[DF['@'] == '@'];

    return DF;


## GRAPHIC ##

def getLastGame(TEAM):
    DF_CURRENT_SEASON = pullDataTeam(LAST_SEASON, TEAM);
    last_game = DF_CURRENT_SEASON.iloc[-1];

    last_game_date = (parse(last_game['Date'])+ datetime.timedelta(days=1)).strftime('%d %B %Y');
    last_game_opponent = last_game['OppN'];
    last_game_team_points = last_game['Tm'];
    last_game_opp_points = last_game['OppP'];

    return last_game_date, last_game_opponent, last_game_team_points, last_game_opp_points;

def currentSeasonBarPlot(TEAM):
    DF_CURRENT_SEASON = pullDataTeam(LAST_SEASON, TEAM);

    differences =  DF_CURRENT_SEASON['Tm'] - DF_CURRENT_SEASON['OppP'];
    len_differences = len(differences);

    colors = ["green" if diff > 0 else "red" for diff in differences];

    fig = plt.figure();
    plt.bar(np.arange(1,len_differences+1),differences,color=colors);
    plt.xlabel("Games (in order)");
    plt.ylabel("Difference in points");
    plt.title(f"CURRENT SEASON {LAST_SEASON-1}/{LAST_SEASON}");
    plt.xlim(0,len_differences+1);
    plt.tick_params(labelbottom = False, bottom = False);
    plt.hlines(0,0,len_differences+1, colors="black", linewidth=1);

    return fig;

def winsLossesPer(DF_PULL_SEASON_RANGE):
    team_games_win_loss = (DF_PULL_SEASON_RANGE['W/L'] == 'W');

    number_of_wins = team_games_win_loss.sum();
    percent_wins = round(number_of_wins/len(team_games_win_loss),2);

    number_of_losses = len(team_games_win_loss) - number_of_wins;
    percent_losses = round(1 - percent_wins,2);

    fig = plt.figure();
    plt.pie([number_of_wins, number_of_losses], labels=[f"Wins {percent_wins}%", f"Losses {percent_losses}%"], colors=["green", "red"]);
    plt.title(f"WINS|LOSSES PERCENTAGES");

    return fig;

def teamPointsPerGame(DF_PULL_SEASON_RANGE, TEAM):
    points_per_game = DF_PULL_SEASON_RANGE['Tm'];

    fig = plt.figure();

    plt.hist(points_per_game, color="#030f20");
    plt.vlines(points_per_game.mean(), 
        plt.ylim()[0], 
        plt.ylim()[1], 
        color="#ff8800", 
        label=f"Mean {round(points_per_game.mean(),2)}"
    );
    plt.title(f"{TEAM} POINTS SCORED PER GAME");
    plt.xlabel("Number of points");
    plt.ylabel("Count");
    plt.legend();

    return fig;

def ooPPointsPerGame(DF_PULL_SEASON_RANGE, TEAM):
    opponent_points_per_game = DF_PULL_SEASON_RANGE['OppP'];

    fig = plt.figure();

    plt.hist(opponent_points_per_game, color="#030f20");
    plt.vlines(opponent_points_per_game.mean(), 
        plt.ylim()[0], 
        plt.ylim()[1], 
        color="#ff8800", 
        label=f"Mean {round(opponent_points_per_game.mean(),2)}"
    );
    plt.title(f"OPPONENTS POINTS SCORED PER GAME AGAINTS {TEAM}");
    plt.xlabel("Number of points");
    plt.ylabel("Count");
    plt.legend();

    return fig;

def pointsScoreBoxPlots(DF_PULL_SEASON_RANGE, TEAM):
    points_per_game = DF_PULL_SEASON_RANGE['Tm'];
    opponent_points_per_game = DF_PULL_SEASON_RANGE['OppP'];

    fig = plt.figure();

    plt.boxplot(
        (points_per_game, opponent_points_per_game), 
        notch=True, sym=".", 
        patch_artist=True, 
        boxprops=dict(facecolor="#030f20"),
        medianprops=dict(color="#ff8800"),
        whiskerprops=dict(color="#030f20"),
        capprops=dict(color="#030f20"));
    plt.title("POINTS SCORED PER GAME");
    plt.xticks(ticks=[1,2],labels=[f"{TEAM}", f"Opponents againts {TEAM}"]);
    plt.ylabel("Points");

    return fig;




