import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import PoissonRegressor
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn import metrics
from tensorflow.keras.models import load_model
from joblib import dump, load

"""
Gls - Goals Scored (not used)
Ast - Assists (not used)
PK - Penalties Scored
PKatt - Penalties Attempted
Sh - Shots
SoT - Shots on Target
CrdY - Yellow Cards
CrdR - Red Cards
Touches - Number of ball touches (not used)
Press - Number of times aplying pressure to opponent
Tkl- Tackles made
Int - Interceptions Made
Blocks - Blocks Made
SCA - Shot Creating Actions
GCA - Goal Creating Actions
Cmp - Passes Completed (not used)
Att - Passes Attempted (not used)
Cmp% - Passing Accuracy
Prog - Progressive Passes
Succ - Succesful dribbles
Att_Dribbles - Attempted Dribles
SoTA - Shots on Target Against (not used)
GA - Goals Against (not used)
Saves - Saves Made (not used)
Save% - Save Percentage
PSxG - Post Shot expected goals (not used)
Possession - Ball Possession

"""

def PK_Accuracy(args):
    if args[1] > 0:
        result = 0 
    else:
        result = args[0]*100.0/args[1]
        
    return args.size


def encodingResults(res):
    if res == 'H':
        return 0
    elif res == 'X':
        return 1
    elif res == 'A':
        return 2

#Configure modules to 
runTeamStatsPlots = False
runGameStatsPlots = False
runPCA = False
runSotModel = False
runSavesModel = True
runTacklesModel = False
runSCAModel = False
runGCAModel = True
runModels = True
runNeuralNetwork = True


## Importing Team Stats Data

#teamStats_df = pd.read_csv("Final Data/EPL_21_22_TeamStats.csv", delimiter=';')
teamStats_df = pd.read_csv("Final Data_New/Bundesliga/CSV/Bundesliga_All_TeamStats.csv", delimiter=';')


## Dropping unused features

teamStats_df.drop('Gls',axis=1,inplace=True)
teamStats_df.drop('Ast',axis=1,inplace=True)
teamStats_df.drop('PK',axis=1,inplace=True)
teamStats_df.drop('PKatt',axis=1,inplace=True)
teamStats_df.drop('Touches',axis=1,inplace=True)
teamStats_df['Prog'] =  teamStats_df['Prog']*100/teamStats_df['Cmp']
teamStats_df.drop('Cmp',axis=1,inplace=True)
teamStats_df.drop('Att',axis=1,inplace=True)
#teamStats_df['Dribble Succ%'] =  teamStats_df['Succ']*100/teamStats_df['Att_dribbles']
#teamStats_df.drop('Succ',axis=1,inplace=True)
#teamStats_df.drop('Att_dribbles',axis=1,inplace=True)
teamStats_df['Cmp%'] = pd.to_numeric(teamStats_df['Cmp%'].str.replace(',','.'))
teamStats_df['Possession'] = teamStats_df['Possession'].str.replace('%','').astype(np.float64)
teamStats_df.drop('GA',axis=1,inplace=True)
teamStats_df.drop('SoTA',axis=1,inplace=True)


##Plots Team Stats
if(runTeamStatsPlots):
    teamStats_df.corr()['GCA'].sort_values().drop('GCA').plot(kind='bar')
    plt.title('GCA correlation')
    
    plt.show()
    
    teamStats_df.corr()['SCA'].sort_values().drop('SCA').drop('GCA').plot(kind='bar')
    plt.title('SCA correlation')
    
    plt.show()
    
    teamStats_df.corr()['SoT'].sort_values().drop('SoT').drop('SCA').drop('GCA').plot(kind='bar')
    plt.title('SoT correlation')
    
    plt.show()
    
    teamStats_df.corr()['CrdY'].sort_values().drop('CrdY').plot(kind='bar')
    plt.title('CrdY correlation')
    
    plt.show()
    
    teamStats_df.corr()['CrdR'].sort_values().drop('CrdR').plot(kind='bar')
    plt.title('CrdR correlation')
    
    plt.show()
    
    teamStats_df.corr()['Press'].sort_values().drop('Press').plot(kind='bar')
    plt.title('Press correlation')
    
    plt.show()
    
    teamStats_df.corr()['Tkl'].sort_values().drop('Tkl').plot(kind='bar')
    plt.title('Tkl correlation')
    
    plt.show()
    
    teamStats_df.corr()['Int'].sort_values().drop('Int').plot(kind='bar')
    plt.title('Int correlation')
    
    plt.show()
    
    teamStats_df.corr()['Blocks'].sort_values().drop('Blocks').plot(kind='bar')
    plt.title('Blocks correlation')
    
    plt.show()
    
    teamStats_df.corr()['Cmp%'].sort_values().drop('Cmp%').plot(kind='bar')
    plt.title('Cmp% correlation')
    
    plt.show()
    
    teamStats_df.corr()['Prog'].sort_values().drop('Prog').plot(kind='bar')
    plt.title('Prog correlation')
    
    plt.show()
    
    teamStats_df.corr()['Saves'].sort_values().drop('Saves').plot(kind='bar')
    plt.title('Saves correlation')
    
    plt.show()
    
    teamStats_df.corr()['Succ'].sort_values().drop('Succ').plot(kind='bar')
    plt.title('Succ correlation')
    
    plt.show()
    
    teamStats_df.corr()['Att_dribbles'].sort_values().drop('Att_dribbles').plot(kind='bar')
    plt.title('Att_dribbles correlation')
    
    plt.show()
    
    sns.scatterplot(data = teamStats_df, x = 'CrdY', y = 'CrdR')
    plt.title('CrdY vs CrdR')
    plt.xlabel('CrdY')
    plt.ylabel('CrdR')
    
    plt.show()
    
    sns.heatmap(teamStats_df.corr(),annot=False,linecolor='black',cmap='Spectral',linewidths=1)

## Import/Create Game Stats Dataframe

#gameStats_df = pd.read_csv("Final Data/EPL_21_22_GameReports.csv", delimiter=';')
#gameStats_df = pd.read_csv("Final Data/Ligue1_21_22 _GameReports.csv", delimiter=';')
gameStats_df = pd.read_csv("Final Data_New/Bundesliga/CSV/Bundesliga_All_GameReports.csv", delimiter=';')

gameStats_df.dropna(axis=0, inplace = True)
gameStats_df['Cmp%_H'] = pd.to_numeric(gameStats_df['Cmp%_H'].str.replace(',','.'))
gameStats_df['Cmp%_A'] = pd.to_numeric(gameStats_df['Cmp%_A'].str.replace(',','.'))
gameStats_df['Save%_H'] = pd.to_numeric(gameStats_df['Save%_H'].str.replace(',','.'))
gameStats_df['Save%_A'] = pd.to_numeric(gameStats_df['Save%_A'].str.replace(',','.'))
gameStats_df['Possession_H'] = gameStats_df['Possession_H'].str.replace('%','').astype(np.float64)
gameStats_df['Possession_A'] = gameStats_df['Possession_A'].str.replace('%','').astype(np.float64)

##Plots Game Stats
if(runGameStatsPlots):

    sns.scatterplot(data = gameStats_df, x = 'GCA_H', y = 'Saves_A')
    plt.title('SoT_H vs Saves_A')
    plt.xlabel('SoT_H')
    plt.ylabel('Saves_A')
    
    plt.show()
    
    sns.scatterplot(data = gameStats_df, x = 'GCA_A', y = 'Saves_H')
    plt.title('SoT_H vs Saves_H')
    plt.xlabel('SoT_H')
    plt.ylabel('Saves_H')
    
    plt.show()
        
    sns.heatmap(gameStats_df.corr(),annot=False,linecolor='black',cmap='Spectral',linewidths=1)

cleaned_gameStats = pd.DataFrame()

#cleaned_gameStats['Shots'] =  gameStats_df['Sh_H'] - gameStats_df['Sh_A']
cleaned_gameStats['SoT'] =  gameStats_df['SoT_H'] - gameStats_df['SoT_A']
cleaned_gameStats['CrdY'] =  gameStats_df['CrdY_H'] - gameStats_df['CrdY_A']
cleaned_gameStats['CrdR'] =  gameStats_df['CrdR_H'] - gameStats_df['CrdR_A']
#cleaned_gameStats['Press'] =  gameStats_df['Press_H'] - gameStats_df['Press_A']
cleaned_gameStats['Tackles'] =  gameStats_df['Tkl_H'] - gameStats_df['Tkl_A']
#cleaned_gameStats['Interceptions'] =  gameStats_df['Int_H'] - gameStats_df['Int_A']
#cleaned_gameStats['Blocks'] =  gameStats_df['Blocks_H'] - gameStats_df['Blocks_A']
cleaned_gameStats['SCA'] =  gameStats_df['SCA_H'] - gameStats_df['SCA_A']
cleaned_gameStats['GCA'] =  gameStats_df['GCA_H'] - gameStats_df['GCA_A']
cleaned_gameStats['Pass Acc'] =  gameStats_df['Cmp%_H'] - gameStats_df['Cmp%_A']
cleaned_gameStats['Prog'] =  (gameStats_df['Prog_H']*100/gameStats_df['Cmp_H']) - (gameStats_df['Prog_A']*100/gameStats_df['Cmp_A'])
#cleaned_gameStats['Dribble Succ%'] =  (gameStats_df['Succ_H']*100/gameStats_df['Att_dribbles_H']) - (gameStats_df['Succ_A']*100/gameStats_df['Att_dribbles_A'])
cleaned_gameStats['Save%'] =  gameStats_df['Save%_H'] - gameStats_df['Save%_A']
cleaned_gameStats['Possession'] =  gameStats_df['Possession_H'] - gameStats_df['Possession_A']

cleaned_gameStats['Result'] =  gameStats_df['Result'].apply(encodingResults)
cleaned_gameStats.dropna(axis=0, inplace=True)

print(cleaned_gameStats.head(5))

#sns.countplot(x=cleaned_gameStats['Result'])

#Plots game stats

#sns.pairplot(data = cleaned_gameStats, hue = 'Result')
#plt.show()

#sns.heatmap(cleaned_gameStats.corr(),annot=True,linecolor='black',cmap='viridis')

## Principal Component Analysis
if (runPCA):
    scaler = StandardScaler()
    scaler.fit(cleaned_gameStats.drop('Result', axis=1))
    scaled_data = scaler.transform(cleaned_gameStats.drop('Result', axis=1))
    pca = PCA(n_components=2)
    pca.fit(scaled_data)
    x_pca = pca.transform(scaled_data)
    
    plt.show()
    plt.scatter(x_pca[:,0],x_pca[:,1],c=cleaned_gameStats['Result'],cmap='plasma')
    plt.xlabel('First principal component')
    plt.ylabel('Second Principal Component')
    
    df_comp = pd.DataFrame(pca.components_,columns=cleaned_gameStats.drop('Result',axis=1).columns)
    plt.show()
    sns.heatmap(df_comp,cmap='plasma',)


if(runSotModel):
    sot_LinearRegressor = LinearRegression()
    
    sot_df = teamStats_df[['SoT', 'Sh']]
    
    sot_X = sot_df[['Sh']]
    sot_y = sot_df['SoT']
    
    sot_X_train, sot_X_test , sot_y_train, sot_y_test = train_test_split(sot_X, sot_y, test_size=0.33)
    
    sot_LinearRegressor.fit(sot_X_train, sot_y_train)
    
    sot_pred_lr = np.round(sot_LinearRegressor.predict(sot_X_test))
    print('SoT')
    print('MAE:', metrics.mean_absolute_error(sot_y_test, sot_pred_lr))
    print('MSE:', metrics.mean_squared_error(sot_y_test, sot_pred_lr))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(sot_y_test, sot_pred_lr)))  
    
    dump(sot_LinearRegressor, 'SoT.cls')
    
if(runSavesModel):
    saves_LinearRegressor =  SVR(kernel = 'linear', C = 0.1, epsilon = 0.5)
    
    
    saves_df_H = gameStats_df[['SoT_H', 'Saves_A']]
    saves_df_H.rename(columns={'SoT_H':'SoT', 'Saves_A':'Saves'}, inplace=True)
    saves_df_A = gameStats_df[['SoT_A', 'Saves_H']]
    saves_df_A.rename(columns={'SoT_A':'SoT', 'Saves_H':'Saves'}, inplace=True)
    
    saves_df = saves_df_H.append(saves_df_A)
    
    saves_X = saves_df[['SoT']]
    saves_y = saves_df['Saves']
    
    saves_X_train, saves_X_test , saves_y_train, saves_y_test = train_test_split(saves_X, saves_y, test_size=0.2)
    
    saves_LinearRegressor.fit(saves_X_train, saves_y_train)
    
    saves_pred_lr = np.round(saves_LinearRegressor.predict(saves_X_test))
    print('Saves')
    print('MAE:', metrics.mean_absolute_error(saves_y_test, saves_pred_lr))
    print('MSE:', metrics.mean_squared_error(saves_y_test, saves_pred_lr))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(saves_y_test, saves_pred_lr)))  
    
    dump(saves_LinearRegressor, 'Saves.cls')
    
if(runTacklesModel):
    tkl_LinearRegressor = LinearRegression()
    
    tkl_df = teamStats_df[['Press', 'Tkl']]
    
    tkl_X = tkl_df[['Press']]
    tkl_y = tkl_df['Tkl']
    
    tkl_X_train, tkl_X_test , tkl_y_train, tkl_y_test = train_test_split(tkl_X, tkl_y, test_size=0.33)
    
    tkl_LinearRegressor.fit(tkl_X_train, tkl_y_train)
    
    tkl_pred_lr = np.round(tkl_LinearRegressor.predict(tkl_X_test))
    print('Tkl')
    print('MAE:', metrics.mean_absolute_error(tkl_y_test, tkl_pred_lr))
    print('MSE:', metrics.mean_squared_error(tkl_y_test, tkl_pred_lr))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(tkl_y_test, tkl_pred_lr)))  

    dump(tkl_LinearRegressor, 'Tkl.cls')
    

if(runSCAModel):
    sh_LinearRegressor = LinearRegression()
    
    sh_df = teamStats_df[['SCA', 'Sh', 'Cmp%', 'Possession', 'Prog']]
    
    sh_X = sh_df.drop('Sh', axis=1)
    sh_y = sh_df['Sh']
    
    sh_X_train, sh_X_test , sh_y_train, sh_y_test = train_test_split(sh_X, sh_y, test_size=0.33)
    
    sh_scaler = MinMaxScaler()
    sh_X_train = sh_scaler.fit_transform(sh_X_train)
    sh_X_test = sh_scaler.transform(sh_X_test)

    sh_LinearRegressor.fit(sh_X_train, sh_y_train)
    
    sh_pred_lr = np.round(sh_LinearRegressor.predict(sh_X_test))
    print('Sh')
    print('MAE:', metrics.mean_absolute_error(sh_y_test, sh_pred_lr))
    print('MSE:', metrics.mean_squared_error(sh_y_test, sh_pred_lr))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(sh_y_test, sh_pred_lr)))  

    dump(sh_LinearRegressor, 'Sh.cls')
    dump(sh_scaler, 'Sh_scaler.save')
    
    
if(runGCAModel):
    gca_LinearRegressor = LinearRegression()    
    
    gca_df_H = gameStats_df[['SCA_H', 'SoT_H', 'Saves_A', 'GCA_H']]
    gca_df_H.rename(columns={'SCA_H':'SCA', 'SoT_H':'SoT', 'Saves_A':'Saves', 'GCA_H':'GCA'}, inplace=True)
    gca_df_A = gameStats_df[['SCA_A', 'SoT_A', 'Saves_H', 'GCA_A']]
    gca_df_A.rename(columns={'SCA_A':'SCA', 'SoT_A':'SoT', 'Saves_H':'Saves', 'GCA_A':'GCA'}, inplace=True)
    
    gca_df = gca_df_H.append(gca_df_A)
    
    gca_X = gca_df.drop('GCA', axis=1)
    gca_y = gca_df['GCA']
    
    gca_X_train, gca_X_test , gca_y_train, gca_y_test = train_test_split(gca_X, gca_y, test_size=0.33)
    
    gca_scaler = MinMaxScaler()
    gca_X_train = gca_scaler.fit_transform(gca_X_train)
    gca_X_test = gca_scaler.transform(gca_X_test)

    gca_LinearRegressor.fit(gca_X_train, gca_y_train)
    
    gca_pred_lr = np.round(gca_LinearRegressor.predict(gca_X_test))
    print('GCA')
    print('MAE:', metrics.mean_absolute_error(gca_y_test, gca_pred_lr))
    print('MSE:', metrics.mean_squared_error(gca_y_test, gca_pred_lr))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(gca_y_test, gca_pred_lr)))     

    dump(gca_LinearRegressor, 'GCA.cls')  
    dump(gca_scaler, 'GCA_scaler.save')
    
    
    
## Train/test split
X = cleaned_gameStats.drop('Result', axis=1)
y = cleaned_gameStats['Result']

X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.33)


#Feature scaling
scalar = MinMaxScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

dump(scalar,'GameStats_scaler.save')

if(runNeuralNetwork):
    NNmodel = Sequential()
    
    NNmodel.add(Dense(units=10,activation='tanh'))
    NNmodel.add(Dropout(0.2))
    NNmodel.add(Dense(units=8,activation='tanh'))
    NNmodel.add(Dropout(0.2))
    NNmodel.add(Dense(units=4,activation='tanh'))
    NNmodel.add(Dropout(0.2))
    NNmodel.add(Dense(3, activation='softmax'))
    
    NNmodel.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=tf.keras.metrics.SparseCategoricalAccuracy())
    
    NNmodel.fit(x=X_train, 
          y=y_train,
          batch_size=16,
          epochs=200,
          validation_data=(X_test, y_test), verbose=1
          )
    
    NNmodel.save('NNmodel.h5')
    
    model_loss = pd.DataFrame(NNmodel.history.history)
    model_loss.plot()
    
    y_pred_NN = np.argmax(NNmodel.predict(X_test), axis=-1)
    
    print('\nNeural Network -------')
    print(confusion_matrix(y_test, y_pred_NN))
    print(classification_report(y_test, y_pred_NN))
    print(accuracy_score(y_test, y_pred_NN)*100)




score_logReg = []
score_rFor = []
score_knn = []
score_svc = []

if (runModels):
    #for i in range(1,100):
    ## Logistic Regression
    logRegClassifier = LogisticRegression()
    logRegClassifier.fit(X_train,y_train)
    
    y_pred_log = logRegClassifier.predict(X_test)
    acc_logReg = accuracy_score(y_test, y_pred_log)
    score_logReg.append(acc_logReg)
    
    ## Random Forest
    randForClassifier = RandomForestClassifier(n_estimators=500)
    randForClassifier.fit(X_train,y_train)
    
    y_pred_rFor = randForClassifier.predict(X_test)
    acc_rFor = accuracy_score(y_test, y_pred_rFor)
    score_rFor.append(acc_rFor)
    
    dump(randForClassifier,'RF_Classifier.cls')
    
    ## K-Nearest Neighbours
    knnClassifier = KNeighborsClassifier(n_neighbors=10)
    knnClassifier.fit(X_train,y_train)
    
    y_pred_knn = knnClassifier.predict(X_test)
    acc_knn = accuracy_score(y_test, y_pred_knn)
    score_knn.append(acc_knn)
    
    ## SVM
    svcClassifier = SVC()
    svcClassifier.fit(X_train,y_train)
    
    y_pred_svc = svcClassifier.predict(X_test)
    acc_svc = accuracy_score(y_test, y_pred_svc)
    score_svc.append(acc_svc)
        
    ## Model evaluation
    print('\nLogistic Regression -------')
    #print(confusion_matrix(y_test, y_pred_log))
    #print(classification_report(y_test, y_pred_log))
    print(np.mean(score_logReg)*100)
    
    
    print('\nRandom Forest -------')
    #print(confusion_matrix(y_test, y_pred_rFor))
    #print(classification_report(y_test, y_pred_rFor))
    print(np.mean(score_rFor)*100)
    
    
    print('\nKNN -------')
    #print(confusion_matrix(y_test, y_pred_knn))
    #print(classification_report(y_test, y_pred_knn))
    print(np.mean(score_knn)*100)
    
    
    print('\nSVM -------')
    #print(confusion_matrix(y_test, y_pred_svc))
    #print(classification_report(y_test, y_pred_svc))
    print(np.mean(score_svc)*100)




