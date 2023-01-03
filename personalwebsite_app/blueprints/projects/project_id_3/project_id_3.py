from flask import Blueprint, render_template, redirect, url_for, request
from personalwebsite_app.jinja2_env import projects_base_structure_jinja2_env
from .nba_teams import *
import base64
from io import BytesIO

project_id_3_bp = Blueprint(name='project_id_3_bp', 
                    import_name=__name__,
                    static_folder='static',
                    template_folder='templates/project_id_3',
                    url_prefix="/project3");

@project_id_3_bp.route('/')
def project3():
    return redirect(url_for('project_id_3_bp.loadFigures', team='ATL'));

@project_id_3_bp.route('/<team>/', methods=['GET','POST'])
def loadFigures(team):

    SEASON_RANGE_START, SEASON_RANGE_END = LAST_SEASON, LAST_SEASON;

    choose_season_error_message = "";
    if request.method == 'POST':
        SEASON_RANGE_START = int(request.form["rangestart"]);
        SEASON_RANGE_END = int(request.form["rangeend"]);

        if SEASON_RANGE_START > SEASON_RANGE_END:
            choose_season_error_message = "Season range start must be less than or equal to Season range end.";
            SEASON_RANGE_START, SEASON_RANGE_END = LAST_SEASON, LAST_SEASON;


    project_base_html_loader = projects_base_structure_jinja2_env.get_template("projects_base_structure.html");

    last_game_date, last_game_opponent, last_game_team_points, last_game_opp_points = getLastGame(team);
    
    DF_PULL_SEASON_RANGE = pullDataTeamSeasonRange(SEASON_RANGE_START, SEASON_RANGE_END, team);
    number_of_games = len(DF_PULL_SEASON_RANGE);

    fig_barplot = currentSeasonBarPlot(team);
    fig_barplot_tmp_file = BytesIO();
    fig_barplot.savefig(fig_barplot_tmp_file, format='png');
    fig_barplot_encoded = base64.b64encode(fig_barplot_tmp_file.getvalue()).decode('utf-8');
    fig_barplot_tmp_file.close();

    fig_wl = winsLossesPer(DF_PULL_SEASON_RANGE);
    fig_wl_tmp_file = BytesIO();
    fig_wl.savefig(fig_wl_tmp_file, format='png');
    fig_wl_encoded  = base64.b64encode(fig_wl_tmp_file.getvalue()).decode('utf-8');
    fig_wl_tmp_file.close();


    fig_points = teamPointsPerGame(DF_PULL_SEASON_RANGE, team);
    fig_points_tmp_file = BytesIO();
    fig_points.savefig(fig_points_tmp_file, format='png');
    fig_points_encoded = base64.b64encode(fig_points_tmp_file.getvalue()).decode('utf-8');
    fig_points_tmp_file.close();


    fig_opp_points = ooPPointsPerGame(DF_PULL_SEASON_RANGE, team);
    fig_opp_points_tmp_file  = BytesIO();
    fig_opp_points.savefig(fig_opp_points_tmp_file, format='png');
    fig_opp_points_encoded = base64.b64encode(fig_opp_points_tmp_file.getvalue()).decode('utf-8');
    fig_opp_points_tmp_file.close();


    fig_points_boxplot = pointsScoreBoxPlots(DF_PULL_SEASON_RANGE, team);
    fig_points_boxplot_tmp_file = BytesIO();
    fig_points_boxplot.savefig(fig_points_boxplot_tmp_file, format='png');
    fig_points_boxplot_encoded = base64.b64encode(fig_points_boxplot_tmp_file.getvalue()).decode('utf-8');
    fig_points_boxplot_tmp_file.close();

    return render_template('project3.html',
                            SEASON_RANGE_START=SEASON_RANGE_START,
                            SEASON_RANGE_END=SEASON_RANGE_END,
                            FIRST_SEASON=FIRST_SEASON,
                            LAST_SEASON=LAST_SEASON,
                            team_acronym=team,
                            opp_acronym=last_game_opponent,
                            choose_season_error_message=choose_season_error_message,
                            last_game_opponent_name=NBA_TEAMS[last_game_opponent][2],
                            last_game_team_name=NBA_TEAMS[team][2],
                            last_game_team_points=int(last_game_team_points), 
                            last_game_opp_points=int(last_game_opp_points),
                            last_game_date=last_game_date,
                            project_base_html_loader=project_base_html_loader,
                            fig_barplot_encoded=fig_barplot_encoded,
                            fig_wl_encoded=fig_wl_encoded,
                            number_of_games=number_of_games,
                            fig_points_encoded=fig_points_encoded,
                            fig_opp_points_encoded=fig_opp_points_encoded,
                            fig_points_boxplot_encoded=fig_points_boxplot_encoded);

