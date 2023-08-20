#Import libraries
import pandas as pd
import requests
from bs4 import BeautifulSoup
import xlsxwriter
import os
import time
from lxml import etree
import re


def scoresfixtures(link,ids):
    '''
    Description: This Class picks all the games that had in one season and combines all links to one especific list
    
    Inputs:
        - link: The link of the main page that have all season games desired.
        - ids: The ID of the championship table
        
    Outputs:
        - specific list that has all the links os all matches of the season
    
    
    '''
    req = requests.get(link)
    if req.status_code == 200:
        content = req.content

    soup = BeautifulSoup(content, 'html.parser')
    tb = soup.find(id=ids)

    s1= []
    s2= []
    for i in tb.find_all("a"):
            s1.append(str(i))
            s2.append(str(i.get_text('href')))


    # Calling DataFrame constructor after zipping 
    # both lists, with columns specified 
    di = pd.DataFrame(list(zip(s1, s2)), 
                   columns =['Codes', 'ID']) 

    s4=[]
    for i in di["Codes"]:
        i = i.replace('<a href="','')
        i = i.replace('</a>','')
        s4.append(str(i))


    s5 = []

    for i in di['Codes']:
        if "matches" in i:
            s5.append(str(i))
        else:
            s5.append(0)

    s6 = []
    for i in di["Codes"]:
        if '<a href="/en/squads/' in i:
            i = i.replace('<a href="/en/squads/','')
            i = i[0:8]
            s6.append(str(i))
        else:
            s6.append(0)        

    # Calling DataFrame constructor after zipping 
    # both lists, with columns specified 
    da = pd.DataFrame(list(zip(s1, s2,s4,s5,s6)), 
                   columns =['CODES', 'ID','URL_FINAL','PARTIDAS_2019',"TEAM_CODE"])        
    
    s9 = []
    for i in da["URL_FINAL"]:
        if 'Match Report' in i:
            s9.append(str(i))
        else:
            pass
    
    return s9

def matchStats(url):
    '''
    Description: This function goes to de URL of the match and treat all data in order to append it in one single Dataframe.
    
    Input:
        - url: Url of the html page
        
    Output:
        - Dataframe treated from the match saved on my machine excel file
    
    '''
    #make the request
    pg = 'https://fbref.com'
    url_pg = pg+ url
    req = requests.get(url_pg)
    if req.status_code == 200:
        content = req.content
    #accessing data from site
    soup = BeautifulSoup(content, 'html.parser')
    dom = etree.HTML(str(soup))

    #table_stats = soup.find_all(class_ = "stats_table")
    table_stats = soup.find_all('table', id = re.compile("._summary"))
    table_team_home = table_stats[0]
    table_team_away = table_stats[1]
    table_time_3 = soup.find(class_='venuetime')
    
    table_gk_stats = soup.find_all('table', id = re.compile("keeper_stats_."))
    table_home_gk = table_gk_stats[0]
    table_away_gk = table_gk_stats[1]
    
    #collecting data
    date = table_time_3.get('data-venue-date') 
    #print(date)

    
    #treating data
    nome = str(soup.title)
    nome = nome.replace(" ","_")
    nome = nome.replace("<title>","")
    nome = nome.replace(".","")
    nome = nome.replace("_Match","")
    nome_final = nome.split("Report")[0]


    #treating data

    #stats_file_name = nome_final.split("_Match")
    stats_file_name = nome_final+ "teamStats"   
    print(stats_file_name)
    
    #report_file_name = nome_final.split("_Match")
    report_file_name = nome_final + "gameReport"   


    # STR transform and reading tables
    table_str_1 = str(table_team_home)
    table_str_2 = str(table_team_away)
    df_1 = pd.read_html(table_str_1)[0]
    df_2 = pd.read_html(table_str_2)[0]
    
    table_str_gk_home = str(table_home_gk)
    table_str_gk_away = str(table_away_gk)
    df_gk_home = pd.read_html(table_str_gk_home)[0]
    df_gk_away = pd.read_html(table_str_gk_away)[0]

    #treating data

    teams = str(nome_final)
    teams = teams.replace("_"," ")
    teams = teams.split(" vs ")
    team_1 = str(teams[0]).strip()
    team_2 = str(teams[1]).strip()
    df_gk_home = df_gk_home["Shot Stopping"]
    df_gk_home['Team'] = team_1
    df_gk_home.set_index('Team') 
    df_gk_away = df_gk_away["Shot Stopping"]
    df_gk_away['Team'] = team_2
    df_gk_away.set_index('Team') 



    home_goals = dom.xpath("((//*[@class='scores'])[1]//following::div)[1]")[0].text
    away_goals = dom.xpath("((//*[@class='scores'])[2]//following::div)[1]")[0].text
    #home_goals = dom.xpath("(//*[@class='score'])[1]")[0].text
    #away_goals = dom.xpath("(//*[@class='score'])[2]")[0].text
    
    if dom.xpath('//*[text()="Possession"]'):
        team_1_possession = dom.xpath('//*[text()="Possession"]//following::strong[1]')[0].text
        team_2_possession = dom.xpath('//*[text()="Possession"]//following::strong[2]')[0].text
    else:
        team_1_possession = 0
        team_2_possession = 0
        
    if dom.xpath('//*[text()="Passing Accuracy"]'):
        team_1_passAcc = dom.xpath('//*[text()="Passing Accuracy"]//following::strong[1]')[0].text
        team_2_passAcc = dom.xpath('//*[text()="Passing Accuracy"]//following::strong[2]')[0].text
    else:
        team_1_passAcc = 0
        team_2_passAcc = 0
    
    
    
    #Dtframe transforming
    df_home = pd.DataFrame(df_1.Performance.drop(df_1.index[0:-1]))
    df_home = df_home.join(df_1.SCA.drop(df_1.index[0:-1]),rsuffix='_SCA')
    df_home = df_home.join(df_1.Passes.drop(df_1.index[0:-1]),rsuffix='_passes')
    df_home = df_home.join(df_1.Dribbles.drop(df_1.index[0:-1]),rsuffix='_dribbles')
    df_home['Date'] = date
    df_home['Possession'] = team_1_possession
    df_home['Team'] = team_1
    df_home['Stadium'] = 'Home'
    df_home.set_index('Team')  
    df_home = df_home.merge(df_gk_home,how="left", suffixes=(False,'_GK'), on='Team')
    #print(df_home)

    df_away = pd.DataFrame(df_2.Performance.drop(df_2.index[0:-1]))
    df_away = df_away.join(df_2.SCA.drop(df_2.index[0:-1]),rsuffix='_SCA')
    df_away = df_away.join(df_2.Passes.drop(df_2.index[0:-1]),rsuffix='_passes')
    df_away = df_away.join(df_2.Dribbles.drop(df_2.index[0:-1]),rsuffix='_dribbles')
    df_away['Date'] = date
    df_away['Possession'] = team_2_possession
    df_away['Team'] = team_2
    df_away['Stadium'] = 'Away'
    df_away.set_index('Team')
    df_away = df_away.merge(df_gk_away,how="left", suffixes=(False,'_GK'), on='Team')
    #print(df_away)

    #APPENDING Dataframes
    
    
    df_gameReport = df_home.join(df_away, lsuffix="_H", rsuffix='_A')
    df_gameReport['Result'] = df_gameReport.apply(lambda a: "H" if home_goals>away_goals else ("A" if home_goals<away_goals else "X"), axis=1 )
    
    if((df_gameReport['Result'] == "H").any()):
        df_home['Result'] = 'W'
        df_away['Result'] = 'L'
    elif((df_gameReport['Result'] == "A").any()):
        df_home['Result'] = 'L'
        df_away['Result'] = 'W'
    elif((df_gameReport['Result'] == "X").any()):
        df_home['Result'] = 'D'
        df_away['Result'] = 'D'
    
    df_game = df_home.append(df_away)
    #print(df_game)
    
    #save excel 
    writer = pd.ExcelWriter("F:/ML_FootballPredictor/TEMP/"+stats_file_name+".xlsx")
    df_game.to_excel(writer,"Stats")
    writer.save()
    
    writer = pd.ExcelWriter("F:/ML_FootballPredictor/TEMP/"+report_file_name+'.xlsx')
    df_gameReport.to_excel(writer,"Stats")
    writer.save()
    
    
def compileAndExportExcel(statsFile, reportFile): 
    '''
    Description: This function goes through all files of the directiory and joins all them in one single dataframe saving in
    excel sheet.
    
    Input:
        - Nome: Name that you want for your excel sheet
        
    Output:
        - Dataframe of all games save as excel sheet
    
    
    '''
    entries = os.listdir("F:/ML_FootballPredictor/TEMP/")
    print(entries)

    teamStats = {}
    gameReports = {}

    teamStats = pd.DataFrame(teamStats)
    gameReports = pd.DataFrame(gameReports)


    for i in entries:
        if "_teamStats.xlsx" in i:
            teamStats = teamStats.append(pd.read_excel("F:/ML_FootballPredictor/TEMP/"+i))
            teamStats = teamStats.drop_duplicates()
            os.remove("F:/ML_FootballPredictor/TEMP/"+i)
        elif "_gameReport.xlsx" in i:
            gameReports = gameReports.append(pd.read_excel("F:/ML_FootballPredictor/TEMP/"+i))
            gameReports = gameReports.drop_duplicates()
            os.remove("F:/ML_FootballPredictor/TEMP/"+i)
        else:
            pass
        
        
    #writer = pd.ExcelWriter("F:/ML_FootballPredictor/Final Data_New/"+statsFile+ ".xlsx")
    teamStats.to_excel("F:/ML_FootballPredictor/Final Data_New/"+statsFile+".xlsx")
    #writer.save()
    
    #writer = pd.ExcelWriter("F:/ML_FootballPredictor/Final Data_New/"+reportFile+ ".xlsx")
    gameReports.to_excel("F:/ML_FootballPredictor/Final Data_New/"+reportFile+".xlsx")
    #writer.save()

#scoresfixtures("https://fbref.com/en/comps/32/10744/schedule/2020-2021-Primeira-Liga-Scores-and-Fixtures","div_sched_3320_1")    



leagueIDs = {"EPL":9,
             "Ligue1":13,
             "Bundesliga":20,
             "SerieA":11,
             "LaLiga":12,
             "Eredivisie":23,
             "LigaPortugal":32,
             "Champions":8,
             "EuropaLeague":19,
             "Conference":882}

#years = ["2017-2018", "2018-2019", "2019-2020", "2020-2021", "2021-2022", "2022-2023"]
#abbrv = ["17_18", "18_19", "19_20", "20_21", "21_22", "22_23"]

years = ["2020-2021", "2021-2022", "2022-2023"]
abbrv = ["20_21", "21_22", "22_23"]

year_cnt = 0

for year in years:
    start = time.time()
    cnt = 0
    
    for url in scoresfixtures("https://fbref.com/en/comps/"+str(leagueIDs['Eredivisie'])+"/"+year+"/schedule/"+year+"-Serie-A-Scores-and-Fixtures","div_sched_"+year+"_"+str(leagueIDs['Eredivisie'])+"_1"):
    #for url in scoresfixtures("https://fbref.com/en/comps/"+str(leagueIDs['EuropaLeague'])+"/"+year+"/schedule/"+year+"-Serie-A-Scores-and-Fixtures","div_sched_all"):
        matchStats(url)
        time.sleep(5)
        cnt = cnt+1
        
        if cnt%5 == 0:
            print(cnt)
        
    
    print('Duration: {} seconds'.format(time.time() - start))
    
    start = time.time()
    compileAndExportExcel("Eredivisie_"+abbrv[year_cnt]+"_TeamStats", "Eredivisie_"+abbrv[year_cnt]+"_GameReports")

    print('Duration: {} seconds'.format(time.time() - start))
    year_cnt = year_cnt+1