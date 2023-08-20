import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.neighbors import KernelDensity
from joblib import dump, load
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import datetime

league = "LigaPortugal"

def encodingResults(res):
    if res == 'H':
        return 0
    elif res == 'X':
        return 1
    elif res == 'A':
        return 2
    
def reverseEncoding(res):
    if res == 0:
        return 'H'
    elif res == 1:
        return 'X'
    elif res == 2:
        return 'A'
    
def encodingForm(res):
    if res == 'W':
        return 1
    elif res == 'L':
        return -1
    elif res == 'D':
        return 0
    
def calculateForm(df):
    form = 0
    if len(df) < 1:
        form = 0
    elif len(df) < 2:
        form = df[-1]*0.075
    elif len(df) < 3:
        form = df[-1]*0.075 + df[-2]*0.025
    elif len(df) < 4:
        form = df[-1]*0.075 + df[-2]*0.025 + df[-3]*0.010
    elif len(df) < 5:
        form = df[-1]*0.075 + df[-2]*0.025 + df[-3]*0.010 + df[-4]*0.005
    else:
        form = df[-1]*0.075 + df[-2]*0.025 + df[-3]*0.010 + df[-4]*0.005 + df[-5]*0.0025
        
    return form

def calculatePoints(df):
    points = 0
    if(len(df)>0):
        for i in range(0,len(df)):
            if df[i] == 1:
                points = points + 3
            elif df[i] == 0:
                points = points + 1
            
    return points
        
   
max_iter = 100
game_iters = 50

## Importing Team Stats Data
teamStats_df = pd.read_csv("F:/ML_FootballPredictor/Final Data_New/" + league + "/CSV/" + league + "_All_TeamStats.csv", delimiter=';')
gameStats_df = pd.read_csv("F:/ML_FootballPredictor/Final Data_New/" + league + "/CSV/" + league + "_All_GameReports.csv", delimiter=';')
teamCoefs_df = pd.read_csv("F:/ML_FootballPredictor/Final Data_New/" + league + "/CSV/" + league + "_teamCoefs.csv", delimiter=';')
teamStats_df['Date'] = pd.to_datetime(teamStats_df['Date'], format='%Y-%m-%d')
gameStats_df['Date_H'] = pd.to_datetime(gameStats_df['Date_H'], format='%Y-%m-%d')
#gameStats_df['Date_H'] = pd.to_datetime(gameStats_df['Date_H'], format='%d/%m/%Y')
gameStats_df['Result'] =  gameStats_df['Result'].apply(encodingResults)
gameStats_df['Goal Diff'] =  gameStats_df.apply((lambda x: x['Gls_H'] - x['Gls_A']), axis=1)
gameStats_df['Save%_H'].fillna(100, inplace=True)
gameStats_df['Save%_A'].fillna(100, inplace=True)
nulls = gameStats_df.isnull().sum()
#gameStats_df.dropna(inplace=True)

teamStats_df.drop('Gls',axis=1,inplace=True)
teamStats_df.drop('Ast',axis=1,inplace=True)
teamStats_df.drop('PK',axis=1,inplace=True)
teamStats_df.drop('PKatt',axis=1,inplace=True)
teamStats_df.drop('Touches',axis=1,inplace=True)
teamStats_df['ShAcc'] = teamStats_df['SoT']/teamStats_df['Sh']
teamStats_df['Prog'] =  teamStats_df['Prog']*100/teamStats_df['Cmp']
teamStats_df.drop('Cmp',axis=1,inplace=True)
teamStats_df.drop('Att',axis=1,inplace=True)
teamStats_df['Cmp%'] = pd.to_numeric(teamStats_df['Cmp%'].str.replace(',','.'))
teamStats_df['Possession'] = teamStats_df['Possession'].str.replace('%','').astype(np.float64)
teamStats_df['Save%'] = pd.to_numeric(teamStats_df['Save%'].str.replace(',','.'))
teamStats_df.drop('GA',axis=1,inplace=True)
teamStats_df.drop('SoTA',axis=1,inplace=True)
teamStats_df['Save%'].fillna(100, inplace=True)
teamStats_df['ShAcc'].fillna(0, inplace=True)
teamStats_df['Result'] = teamStats_df['Result'].apply(encodingForm)
#teamStats_df['Press'].fillna(0, inplace=True)
#teamStats_df.dropna(inplace=True)

teamCoefs_df['att_coef'] = pd.to_numeric(teamCoefs_df['att_coef'].str.replace(',','.'))
teamCoefs_df['def_coef'] = pd.to_numeric(teamCoefs_df['def_coef'].str.replace(',','.'))
teamCoefs_df['Result'] = pd.to_numeric(teamCoefs_df['Result'].str.replace(',','.'))

print(teamStats_df['Team'].unique())
## Importing Models
SoT_model = load('SoT.cls')  #Sh
Saves_model = load('Saves.cls') #SoT
#Tkl_model = load('Tkl.cls')  #Press
Sh_model = load('Sh.cls')  #SoT, Sh, Cmp%, Possession, Prog
Sh_scaler = load('Sh_scaler.save')
GCA_model = load('GCA.cls')  #SCA, SoT, Sh, Cmp%
GCA_scaler = load('GCA_scaler.save')
GameStats_scaler = load('GameStats_scaler.save')
RF_classifier = load('RF_Classifier.cls')
NN_model = load_model('NNmodel.h5')

## Fixtures
gameStats_df['Match'] = gameStats_df.apply((lambda x: str(x["Team_H"]) + '/' + str(x["Team_A"]) + '/' + str(x['Date_H'])), axis=1)

fixtures= gameStats_df['Match']

#fixtures = ["Portimonense/Santa Clara", "Vizela/Mar�timo", "Rio Ave/Pa�os de Ferreira", "Braga/Boavista", 
#            "Chaves/Arouca", "Benfica/Sporting CP", "Porto/Famalic�o", 
#            "Estoril/Casa Pia", "Gil Vicente FC/Vit�ria Guimar�es"]

#fixtures = ["M�nchengladbach/Stuttgart", "Augsburg/Eintracht Frankfurt", "Dortmund/Bochum", "Hertha BSC/Bayern Munich", 
#            "Hoffenheim/RB Leipzig", "Mainz 05/Wolfsburg", "Werder Bremen/Schalke 04", 
#            "Bayer Leverkusen/Union Berlin", "Freiburg/K�ln"]


#fixtures = ["Troyes/Auxerre", "Ajaccio/Strasbourg", "Angers/Lens", "Lorient/Paris Saint-Germain", 
#            "Clermont Foot/Montpellier", "Nice/Brest", "Reims/Nantes", 
#            "Toulouse/Monaco", "Lille/Rennes" , "Marseille/Lyon"]



#fixtures = ["Udinese/Lecce", "Empoli/Sassuolo", "Salernitana/Cremonese", "Atalanta/Napoli", 
#            "Milan/Spezia", "Bologna/Torino", "Monza/Hellas Verona", 
#            "Sampdoria/Fiorentina", "Roma/Lazio" , "Juventus/Internazionale"]



#fixtures = ["Leeds United/Bournemouth", "Manchester City/Fulham", "Nottingham Forest/Brentford", "Wolverhampton Wanderers/Brighton &amp; Hove Albion", 
#            "Everton/Leicester City", "Chelsea/Arsenal", "Aston Villa/Manchester United",
#            "Southampton/Newcastle United", "West Ham United/Crystal Palace", "Tottenham Hotspur/Liverpool"]

#fixtures = ["Girona/Athletic Club", "Getafe/C�diz", "Valladolid/Elche", "Celta Vigo/Osasuna", 
#           "Barcelona/Almer�a", "Atl�tico Madrid/Espanyol", "Real Sociedad/Valencia", 
#           "Villarreal/Mallorca", "Real Betis/Sevilla" , "Rayo Vallecano/Real Madrid"]






							
print(len(fixtures))
#fixtures = ["Brighton &amp; Hove AlbionvsLeicester City", "Manchester UnitedvsArsenal"]


## Generate probability distributions
final_result_rf = []
final_result_nn = []

cnt = 0

for match in fixtures:
    print(match)
    home_team = match.split("/")[0]
    away_team = match.split("/")[1]
    date = match.split("/")[2]
    #date = datetime.datetime.now().strftime("%Y-%m-%d")
    cutoff_date = pd.to_datetime(date).replace(day=1).replace(year=pd.to_datetime(date).year-1)
    if (pd.to_datetime(date).month >= 8):
        season_start_date = pd.to_datetime(date).replace(day=1).replace(month=8).replace(year=pd.to_datetime(date).year)
    else:
        season_start_date = pd.to_datetime(date).replace(day=1).replace(month=8).replace(year=pd.to_datetime(date).year-1)
    #print(season_start_date)
    
    home_att_coef = teamCoefs_df[(teamCoefs_df['Team'] == home_team)]['att_coef'].values[0]
    #print("Home Att coef: " + str(home_att_coef))
    home_def_coef = teamCoefs_df[(teamCoefs_df['Team'] == home_team)]['def_coef'].values[0]
    #print("Home Def coef: " + str(home_def_coef))
    away_att_coef = teamCoefs_df[(teamCoefs_df['Team'] == away_team)]['att_coef'].values[0]
    #print("Away Att coef: " + str(away_att_coef))
    away_def_coef = teamCoefs_df[(teamCoefs_df['Team'] == away_team)]['def_coef'].values[0]
    #print("Away Def coef: " + str(away_def_coef))
    home_ovr_coef = teamCoefs_df[(teamCoefs_df['Team'] == home_team)]['Result'].values[0]
    #print(home_ovr_coef)
    away_ovr_coef = teamCoefs_df[(teamCoefs_df['Team'] == away_team)]['Result'].values[0]
    #print(away_ovr_coef)
    
    #home_att_coef = (home_att_coef/away_def_coef)[0]
    #print("Home Att coef: " + str(home_att_coef))
    #away_att_coef = (away_att_coef/home_def_coef)[0]
    #print("Away Att coef: " + str(away_att_coef))
    
    home_game_dif = 0.2*((home_att_coef-away_def_coef)+(home_def_coef-away_att_coef)+(home_ovr_coef-away_ovr_coef))
    away_game_dif = 0.2*((-home_att_coef+away_def_coef)+(-home_def_coef+away_att_coef)+(-home_ovr_coef+away_ovr_coef))
    #print(home_game_dif)
    #print(away_game_dif)
        
    print(date)
    
    home_games = teamStats_df[(teamStats_df['Team'] == home_team) & (teamStats_df['Date'] < date) & (teamStats_df['Date'] >= cutoff_date)].sort_values('Date')['Result']
    away_games = teamStats_df[(teamStats_df['Team'] == away_team) & (teamStats_df['Date'] < date) & (teamStats_df['Date'] >= cutoff_date)].sort_values('Date')['Result']
    #print(home_games)
    #print(away_games)
    
    home_season_games = teamStats_df[(teamStats_df['Team'] == home_team) & (teamStats_df['Date'] < date) & (teamStats_df['Date'] >= season_start_date)].sort_values('Date')['Result']
    away_season_games = teamStats_df[(teamStats_df['Team'] == away_team) & (teamStats_df['Date'] < date) & (teamStats_df['Date'] >= season_start_date)].sort_values('Date')['Result']
    
    home_points = calculatePoints(home_season_games.values)
    #print(home_points)
    away_points = calculatePoints(away_season_games.values)
    #print(home_points/away_points)
    
    home_form_ovr = 1+calculateForm(home_games.values)+home_game_dif
    away_form_ovr = 1+calculateForm(away_games.values)+away_game_dif
    #print("Home form: " + str(home_form_ovr))
    #print("Away form: " + str(away_form_ovr))
    
    #& (teamStats_df['Stadium'] == 'Home') 
    #& (teamStats_df['Stadium'] == 'Away') 
    # Shots
    """
    home_shots = np.round(pd.DataFrame(teamStats_df[(teamStats_df['Team'] == home_team) & (teamStats_df['Date'] <= date) & (teamStats_df['Date'] >= cutoff_date)]['Sh'])*home_form_ovr*home_game_ovr)    
    home_shots_model = KernelDensity()
    home_shots_model.fit(home_shots)    

    away_shots = np.round(pd.DataFrame(teamStats_df[(teamStats_df['Team'] == away_team) & (teamStats_df['Date'] <= date) & (teamStats_df['Date'] >= cutoff_date)]['Sh'])*away_form_ovr*away_game_ovr)
    away_shots_model = KernelDensity()
    away_shots_model.fit(away_shots) 
    """
    
    #SCA
    home_SCA = np.round(pd.DataFrame(teamStats_df[(teamStats_df['Team'] == home_team) & (teamStats_df['Date'] <= date) & (teamStats_df['Date'] >= cutoff_date)]['SCA'])*home_form_ovr)    
    home_SCA_model = KernelDensity()
    home_SCA_model.fit(home_SCA)    

    away_SCA = np.round(pd.DataFrame(teamStats_df[(teamStats_df['Team'] == away_team) & (teamStats_df['Date'] <= date) & (teamStats_df['Date'] >= cutoff_date)]['SCA'])*away_form_ovr)
    away_SCA_model = KernelDensity()
    away_SCA_model.fit(away_SCA) 
    
    
    # Shooting Accuracy
    home_shAcc = np.round(pd.DataFrame(teamStats_df[(teamStats_df['Team'] == home_team) & (teamStats_df['Date'] >= cutoff_date)]['ShAcc'])*home_form_ovr)    
    home_shAcc_model = KernelDensity()
    home_shAcc_model.fit(home_shAcc)    

    away_shAcc = np.round(pd.DataFrame(teamStats_df[(teamStats_df['Team'] == away_team) & (teamStats_df['Date'] >= cutoff_date)]['ShAcc'])*away_form_ovr)
    away_shAcc_model = KernelDensity()
    away_shAcc_model.fit(away_shAcc) 
    
    # Yellow Cards
    home_crdY = pd.DataFrame(teamStats_df[(teamStats_df['Team'] == home_team) & (teamStats_df['Date'] <= date) & (teamStats_df['Date'] >= cutoff_date)]['CrdY'])    
    home_crdY_model = KernelDensity()
    home_crdY_model.fit(home_crdY)
    
    away_crdY = pd.DataFrame(teamStats_df[(teamStats_df['Team'] == away_team) & (teamStats_df['Date'] <= date) & (teamStats_df['Date'] >= cutoff_date)]['CrdY'])    
    away_crdY_model = KernelDensity()
    away_crdY_model.fit(away_crdY) 
    
    # Red Cards
    home_crdR = pd.DataFrame(teamStats_df[(teamStats_df['Team'] == home_team) & (teamStats_df['Date'] <= date) & (teamStats_df['Date'] >= cutoff_date)]['CrdR'])    
    home_crdR_model = KernelDensity()
    home_crdR_model.fit(home_crdR)
    
    away_crdR = pd.DataFrame(teamStats_df[(teamStats_df['Team'] == away_team) & (teamStats_df['Date'] <= date) & (teamStats_df['Date'] >= cutoff_date)]['CrdR'])    
    away_crdR_model = KernelDensity()
    away_crdR_model.fit(away_crdR)
    
    # Tackles
    home_tackles = np.round(pd.DataFrame(teamStats_df[(teamStats_df['Team'] == home_team) & (teamStats_df['Date'] <= date) & (teamStats_df['Date'] >= cutoff_date)]['Tkl'])*home_form_ovr)    
    home_tackles_model = KernelDensity()
    home_tackles_model.fit(home_tackles)
    
    away_tackles = np.round(pd.DataFrame(teamStats_df[(teamStats_df['Team'] == away_team) & (teamStats_df['Date'] <= date) & (teamStats_df['Date'] >= cutoff_date)]['Tkl'])*away_form_ovr)    
    away_tackles_model = KernelDensity()
    away_tackles_model.fit(away_tackles)
    
    # Passing Accuracy
    home_cmp = np.round(pd.DataFrame(teamStats_df[(teamStats_df['Team'] == home_team) & (teamStats_df['Date'] <= date) & (teamStats_df['Date'] >= cutoff_date)]['Cmp%'])*home_form_ovr)   
    home_cmp_model = KernelDensity()
    home_cmp_model.fit(home_cmp)
    
    away_cmp = np.round(pd.DataFrame(teamStats_df[(teamStats_df['Team'] == away_team) & (teamStats_df['Date'] <= date) & (teamStats_df['Date'] >= cutoff_date)]['Cmp%'])*away_form_ovr)    
    away_cmp_model = KernelDensity()
    away_cmp_model.fit(away_cmp)
    
    # Progressive Passes
    home_prog = np.round(pd.DataFrame(teamStats_df[(teamStats_df['Team'] == home_team) & (teamStats_df['Date'] <= date) & (teamStats_df['Date'] >= cutoff_date)]['Prog'])*home_form_ovr)    
    home_prog_model = KernelDensity()
    home_prog_model.fit(home_prog)
    
    away_prog = np.round(pd.DataFrame(teamStats_df[(teamStats_df['Team'] == away_team) & (teamStats_df['Date'] <= date) & (teamStats_df['Date'] >= cutoff_date)]['Prog'])*away_form_ovr)    
    away_prog_model = KernelDensity()
    away_prog_model.fit(away_prog)
    
    # Save Percentage
    home_save = np.round(pd.DataFrame(teamStats_df[(teamStats_df['Team'] == home_team) & (teamStats_df['Date'] <= date) & (teamStats_df['Date'] >= cutoff_date)]['Save%'])*home_form_ovr)    
    home_save_model = KernelDensity()
    home_save_model.fit(home_save)
    
    away_save = np.round(pd.DataFrame(teamStats_df[(teamStats_df['Team'] == away_team) & (teamStats_df['Date'] <= date) & (teamStats_df['Date'] >= cutoff_date)]['Save%'])*away_form_ovr)   
    away_save_model = KernelDensity()
    away_save_model.fit(away_save)
    
    # Ball Possession
    home_poss = np.round(pd.DataFrame(teamStats_df[(teamStats_df['Team'] == home_team) & (teamStats_df['Date'] <= date) & (teamStats_df['Date'] >= cutoff_date)]['Possession'])*home_form_ovr)
    away_poss = np.round(pd.DataFrame(teamStats_df[(teamStats_df['Team'] == away_team) & (teamStats_df['Date'] <= date) & (teamStats_df['Date'] >= cutoff_date)]['Possession'])*away_form_ovr)  
    poss_model = KernelDensity()
    if(len(home_poss) > len(away_poss)):
        home_poss=home_poss[:len(away_poss)]
    elif(len(home_poss) < len(away_poss)):
        away_poss=away_poss[:len(home_poss)]
    poss_model.fit(home_poss.values - away_poss.values)

    rf_predicted_results = np.array([0,0,0])
    nn_predicted_results = np.array([0,0,0])
    
    for j in range(0,game_iters):
        home_poss_stat = 0
        home_SCA_stat = 0
        home_ShAcc_stat = 0
        home_tackles_stat = 0
        home_cmp_stat = 0
        home_prog_stat = 0
        home_SavePerc_stat = 0
        home_crdY_stat = 0
        home_crdR_stat = 0
        
        away_poss_stat = 0
        away_SCA_stat = 0
        away_ShAcc_stat = 0
        away_tackles_stat = 0
        away_cmp_stat = 0
        away_prog_stat = 0
        away_SavePerc_stat = 0
        away_crdY_stat = 0
        away_crdR_stat = 0
        
        for i in range(0,max_iter):
            #home_form = random.gauss(1,0.12)
            #away_form = random.gauss(1,0.12)
            poss_stat = np.round(poss_model.sample(1))[0,0]
            home_poss_stat = home_poss_stat + 50 + poss_stat/2
            away_poss_stat = away_poss_stat + 50 - poss_stat/2
            
            home_SCA_stat = home_SCA_stat + np.round(home_SCA_model.sample(1))[0,0]
            away_SCA_stat = away_SCA_stat + np.round(away_SCA_model.sample(1))[0,0]
            
            home_ShAcc_stat = home_ShAcc_stat + home_shAcc_model.sample(1)[0,0]
            away_ShAcc_stat = away_ShAcc_stat + away_shAcc_model.sample(1)[0,0]
            
            home_tackles_stat = home_tackles_stat + np.round(home_tackles_model.sample(1))[0,0]
            away_tackles_stat = away_tackles_stat + np.round(away_tackles_model.sample(1))[0,0]
            
            home_cmp_stat = home_cmp_stat + home_cmp_model.sample(1)[0,0]
            away_cmp_stat = away_cmp_stat + away_cmp_model.sample(1)[0,0]
            
            home_prog_stat = home_prog_stat + np.round(home_prog_model.sample(1))[0,0]
            away_prog_stat = away_prog_stat + np.round(away_prog_model.sample(1))[0,0]
            
            home_SavePerc_stat = home_SavePerc_stat + (home_save_model.sample(1))[0,0]
            away_SavePerc_stat = away_SavePerc_stat + (away_save_model.sample(1))[0,0]
            
            home_crdY_stat = home_crdY_stat + np.round(home_crdY_model.sample(1))[0,0]
            away_crdY_stat = away_crdY_stat + np.round(away_crdY_model.sample(1))[0,0]
            
            home_crdR_stat = home_crdR_stat + np.round(home_crdR_model.sample(1))[0,0]
            away_crdR_stat = away_crdR_stat + np.round(away_crdR_model.sample(1))[0,0]
         
            
        home_poss_stat = np.round(home_poss_stat/max_iter)
        home_SCA_stat = np.round(home_SCA_stat/max_iter)
        home_ShAcc_stat = home_ShAcc_stat/max_iter
        if home_ShAcc_stat < 0:
            home_ShAcc_stat = 0
        elif home_ShAcc_stat > 1:
            home_ShAcc_stat = 1
        home_tackles_stat = np.round(home_tackles_stat/max_iter)
        home_cmp_stat = home_cmp_stat/max_iter
        home_prog_stat = np.round(home_prog_stat/max_iter)
        home_SavePerc_stat = home_SavePerc_stat/max_iter
        if home_SavePerc_stat > 100:
            home_SavePerc_stat = 100.0
        home_crdY_stat = np.round(home_crdY_stat/max_iter)
        home_crdR_stat = np.round(home_crdR_stat/max_iter)
        
        away_poss_stat = np.round(away_poss_stat/max_iter)
        away_SCA_stat = np.round(away_SCA_stat/max_iter)
        away_ShAcc_stat = away_ShAcc_stat/max_iter
        if away_ShAcc_stat < 0:
            away_ShAcc_stat = 0
        elif away_ShAcc_stat > 1:
            away_ShAcc_stat = 1
        away_tackles_stat = np.round(away_tackles_stat/max_iter)
        away_cmp_stat = away_cmp_stat/max_iter
        away_prog_stat = np.round(away_prog_stat/max_iter)
        away_SavePerc_stat = away_SavePerc_stat/max_iter
        if away_SavePerc_stat > 100:
            away_SavePerc_stat = 100.0
        away_crdY_stat = np.round(away_crdY_stat/max_iter)
        away_crdR_stat = np.round(away_crdR_stat/max_iter)
        
        scaled_Sh_home_df = Sh_scaler.transform([[home_SCA_stat,home_cmp_stat,home_poss_stat,home_prog_stat]])
        scaled_Sh_away_df = Sh_scaler.transform([[away_SCA_stat,away_cmp_stat,away_poss_stat,away_prog_stat]])
        home_Sh_stat = np.round(Sh_model.predict(scaled_Sh_home_df))[0]
        away_Sh_stat = np.round(Sh_model.predict(scaled_Sh_away_df))[0]
         
        home_SoT_stat = np.round(home_Sh_stat*home_ShAcc_stat)
        away_SoT_stat = np.round(away_Sh_stat*away_ShAcc_stat)
        
        #home_Saves_stat = np.round(Saves_model.predict([[away_SoT_stat]]))[0]
        #away_Saves_stat = np.round(Saves_model.predict([[home_SoT_stat]]))[0]
        
        home_Saves_stat = np.round(away_SoT_stat*home_SavePerc_stat/100)
        away_Saves_stat = np.round(home_SoT_stat*away_SavePerc_stat/100)
        
        #home_SavePerc_stat = home_Saves_stat*100/away_SoT_stat if away_SoT_stat>0 else 100.0
        #away_SavePerc_stat = away_Saves_stat*100/home_SoT_stat if home_SoT_stat>0 else 100.0
        
        scaled_GCA_home_df = GCA_scaler.transform([[home_SCA_stat,home_SoT_stat,away_Saves_stat]])
        scaled_GCA_away_df = GCA_scaler.transform([[away_SCA_stat,away_SoT_stat,home_Saves_stat]])
        home_GCA_stat = np.round(GCA_model.predict(scaled_GCA_home_df))[0]
        away_GCA_stat = np.round(GCA_model.predict(scaled_GCA_away_df))[0]
            
        SoT_stat = home_SoT_stat - away_SoT_stat
        Tkl_stat = home_tackles_stat - away_tackles_stat
        SCA_stat = home_SCA_stat - away_SCA_stat
        GCA_stat = home_GCA_stat - away_GCA_stat
        cmp_stat = home_cmp_stat - away_cmp_stat
        prog_stat = home_prog_stat - away_prog_stat
        save_stat = home_SavePerc_stat - away_SavePerc_stat
        crdY_stat = home_crdY_stat - away_crdY_stat
        crdR_stat = home_crdR_stat - away_crdR_stat
        poss_stat = home_poss_stat - away_poss_stat
        
        game_stats_df = GameStats_scaler.transform([[SoT_stat, crdY_stat, crdR_stat, Tkl_stat, SCA_stat, GCA_stat, cmp_stat, prog_stat, save_stat, poss_stat]])
            
        rf_prediction = RF_classifier.predict(game_stats_df)[0]
        rf_predicted_results[int(rf_prediction)] = rf_predicted_results[rf_prediction] + 1
        #rf_prediction_prob = RF_classifier.predict_proba(game_stats_df)
        #print(rf_prediction_prob[:,rf_prediction][0])
        
        NN_prediction = np.argmax(NN_model.predict(game_stats_df), axis=-1)
        nn_predicted_results[int(NN_prediction)] = nn_predicted_results[NN_prediction] + 1
        
    #print(rf_predicted_results)
    predicted_rf_final_result = rf_predicted_results.argmax()
    #print(predicted_rf_final_result)
    home_rf_prob = rf_predicted_results[0]/game_iters
    #print(home_rf_prob)
    draw_rf_prob = rf_predicted_results[1]/game_iters
    #print(draw_rf_prob)
    away_rf_prob = rf_predicted_results[2]/game_iters 
    #print(away_rf_prob)
    
    if((home_rf_prob >= draw_rf_prob and home_rf_prob <= away_rf_prob) or (home_rf_prob <= draw_rf_prob and home_rf_prob >= away_rf_prob)):
        second_prediction_rf = 0
    elif((draw_rf_prob >= home_rf_prob and draw_rf_prob <= away_rf_prob) or (draw_rf_prob <= home_rf_prob and draw_rf_prob >= away_rf_prob)):
        second_prediction_rf = 1
    else:
        second_prediction_rf = 2
        
    final_result_rf.append([home_team,away_team,predicted_rf_final_result,second_prediction_rf,home_rf_prob,draw_rf_prob,away_rf_prob])
    
    #print(nn_predicted_results)  
    predicted_nn_final_result = nn_predicted_results.argmax()
    #print(predicted_rf_final_result)
    home_nn_prob = nn_predicted_results[0]/game_iters
    #print(home_rf_prob)
    draw_nn_prob = nn_predicted_results[1]/game_iters
    #print(draw_rf_prob)
    away_nn_prob = nn_predicted_results[2]/game_iters
    
    if((home_nn_prob >= draw_nn_prob and home_nn_prob <= away_nn_prob) or (home_nn_prob <= draw_nn_prob and home_nn_prob >= away_nn_prob)):
        second_prediction_nn = 0
    elif((draw_nn_prob >= home_nn_prob and draw_nn_prob <= away_nn_prob) or (draw_nn_prob <= home_nn_prob and draw_nn_prob >= away_nn_prob)):
        second_prediction_nn = 1
    else:
        second_prediction_nn = 2
    
    final_result_nn.append([home_team,away_team,predicted_nn_final_result,second_prediction_nn,home_nn_prob,draw_nn_prob,away_nn_prob])
    

finalResult_rf_df = pd.DataFrame(final_result_rf, columns=['Home','Away','Predicted Result','Second Predicted Result','Home win prob', 'Draw prob', 'Away win prob'])
finalResult_nn_df = pd.DataFrame(final_result_nn, columns=['Home','Away','Predicted Result','Second Predicted Result','Home win prob', 'Draw prob', 'Away win prob'])
print(finalResult_rf_df.head())
print(finalResult_nn_df.head())


print("========RANDOM FOREST=======")
print(classification_report(gameStats_df['Result'], finalResult_rf_df['Predicted Result']))
print(confusion_matrix(gameStats_df['Result'], finalResult_rf_df['Predicted Result']))

print(classification_report(gameStats_df['Result'], finalResult_rf_df['Second Predicted Result']))
print(confusion_matrix(gameStats_df['Result'], finalResult_rf_df['Second Predicted Result']))

finalResult_rf_df['Goal Diff'] = gameStats_df['Goal Diff']
finalResult_rf_df['Correct Prediction'] = gameStats_df['Result']
finalResult_rf_df['Correct Prediction BIN'] = finalResult_rf_df.apply((lambda x : 1 if(x['Predicted Result'] == x['Correct Prediction']) else 0), axis=1)
finalResult_rf_df['Date'] = gameStats_df['Date_H']
sns.countplot(x=finalResult_rf_df['Goal Diff'], hue=finalResult_rf_df['Correct Prediction BIN'])
plt.title("Random Forest")
plt.show()
sns.countplot(x=finalResult_rf_df['Predicted Result'], hue=finalResult_rf_df['Correct Prediction BIN'])
plt.title("Random Forest")
plt.show()


"""
finalResult_rf_df['Predicted Result'] = finalResult_rf_df['Predicted Result'].apply(reverseEncoding)
#finalResult_rf_df['Second Predicted Result'] = finalResult_rf_df['Second Predicted Result'].apply(reverseEncoding)
writer = pd.ExcelWriter(league + '_RF_PredResults.xlsx')
finalResult_rf_df.to_excel(writer,"Predictions")
writer.save()
"""

print("========NEURAL NETWORK=======")
print(classification_report(gameStats_df['Result'], finalResult_nn_df['Predicted Result']))
print(confusion_matrix(gameStats_df['Result'], finalResult_nn_df['Predicted Result']))

print(classification_report(gameStats_df['Result'], finalResult_nn_df['Second Predicted Result']))
print(confusion_matrix(gameStats_df['Result'], finalResult_nn_df['Second Predicted Result']))

finalResult_nn_df['Goal Diff'] = gameStats_df['Goal Diff']
finalResult_nn_df['Correct Prediction'] = gameStats_df['Result']
finalResult_nn_df['Correct Prediction BIN'] = finalResult_nn_df.apply((lambda x : 1 if(x['Predicted Result'] == x['Correct Prediction']) else 0), axis=1)
finalResult_nn_df['Date'] = gameStats_df['Date_H']
sns.countplot()
plt.title("Neural Network")
plt.show()
sns.countplot(x=finalResult_nn_df['Predicted Result'], hue=finalResult_nn_df['Correct Prediction BIN'])
plt.title("Neural Network")
plt.show()

"""
finalResult_nn_df['Predicted Result'] = finalResult_nn_df['Predicted Result'].apply(reverseEncoding)
#finalResult_nn_df['Second Predicted Result'] = finalResult_nn_df['Second Predicted Result'].apply(reverseEncoding)
writer = pd.ExcelWriter(league + '_NN_PredResults.xlsx')
finalResult_nn_df.to_excel(writer,"Predictions")
writer.save()
"""



    



