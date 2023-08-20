import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.preprocessing import MinMaxScaler
import datetime

def encodingForm(res):
    if res == 'W':
        return 1
    elif res == 'L':
        return -1
    elif res == 'D':
        return 0
    
def powerScore(df):
    score = 0;
    
    for i in range(len(df)):
        if i < 15:
            score = score + df[i]*1
        elif i < 30:
            score = score + df[i]*0.9
        elif i < 40:
            score = score + df[i]*0.75
        elif i < 50:
            score = score + df[i]*0.7
        elif i < 60:
            score = score + df[i]*0.6
        else:
            score = score + df[i]*0.5
            
    return (score/len(df)+1)
    
    
league = "LigaPortugal"
pathXLS = "F:/ML_FootballPredictor/Final Data_New/" + league + "\\XLS"

teamStats_df = pd.read_csv("F:/ML_FootballPredictor/Final Data_New/" + league + "/CSV/" + league + "_All_TeamStats.csv", delimiter=';', encoding='cp1252')
gameStats_df = pd.read_csv("F:/ML_FootballPredictor/Final Data_New/" + league + "/CSV/" + league + "_All_GameReports.csv", delimiter=';', encoding='cp1252')


teamStats_df.drop('PK',axis=1,inplace=True)
teamStats_df.drop('Touches',axis=1,inplace=True)
#teamStats_df.drop('Date',axis=1,inplace=True)
teamStats_df['Prog'] =  teamStats_df['Prog']*100/teamStats_df['Cmp']
teamStats_df['Cmp%'] = (pd.to_numeric(teamStats_df['Cmp%'].str.replace(',','.')))/10
teamStats_df['Possession'] = (teamStats_df['Possession'].str.replace('%','').astype(np.float64))/10
teamStats_df['Save%'] = (pd.to_numeric(teamStats_df['Save%'].str.replace(',','.')))/10
teamStats_df['Save%'].fillna(10, inplace=True)
teamStats_df['Result'] = teamStats_df['Result'].apply(encodingForm)

#coefs = teamStats_df.groupby(['Team','Stadium']).mean()


teams = teamStats_df['Team'].unique()

for team in teams:
    df = teamStats_df[(teamStats_df['Team'] == team)].sort_values('Date')['Result']
    #print(df.head())
    ovr_coef = powerScore(df.values)
    teamStats_df.loc[teamStats_df.Team == team,['Result']] = ovr_coef
    #print(team + ": " + str(ovr_coef))
    
coefs = teamStats_df.groupby(['Team']).mean()

##====================================================
## Attack - + Gls, PKatt, Sh, SoT, SCA, GCA, Prog, Possession - crdR

att_coef = (coefs['Gls']/coefs['SoT'])*10 + coefs['PKatt'] + coefs['SCA']/10 + coefs['GCA'] + coefs['Prog']/10 + coefs['Possession']

Att_maxmin = np.array([[att_coef.values.max(), 1],
                       [att_coef.values.min(),1]])
norm = np.array([0.2,-0.2])

x = np.linalg.solve(Att_maxmin, norm)

att_coef = att_coef*x[0]+x[1]

att_coef = pd.DataFrame(att_coef, columns=['att_coef'])
##====================================================


##====================================================
## Defense - GA, SoTA, CrdR + CrdY, Press, Tkl, Int, Blocks, Saves, Saves%

def_coef = - 10*coefs['CrdR'] - 10*(coefs['SoTA']) + coefs['CrdY'] + coefs['Tkl']/10 + coefs['Int']/10 + coefs['Blocks']/10  

Def_maxmin = np.array([[def_coef.values.max(), 1],
                       [def_coef.values.min(),1]])
norm = np.array([0.2,-0.2])

x = np.linalg.solve(Def_maxmin, norm)

def_coef = def_coef*x[0]+x[1]

def_coef = pd.DataFrame(def_coef, columns=['def_coef'])

##====================================================


ovr_coef = coefs['Result']

Ovr_maxmin = np.array([[ovr_coef.values.max(), 1],
                       [ovr_coef.values.min(),1]])
norm = np.array([0.2,-0.2])

x = np.linalg.solve(Ovr_maxmin, norm)

ovr_coef = ovr_coef*x[0]+x[1]




final_coefs = att_coef.join(def_coef)
final_coefs = final_coefs.join(ovr_coef)


writer = pd.ExcelWriter(pathXLS + "/" + league + '_teamCoefs.xlsx')
final_coefs.to_excel(writer,"Coefs", merge_cells=False)
writer.save()