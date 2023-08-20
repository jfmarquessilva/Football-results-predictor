import xlsxwriter
import os
import pandas as pd


leagueIDs = ['LigaPortugal']#"EPL","Ligue1","Bundesliga","SerieA","LaLiga"]#,"Eredivisie",
             #"LigaPortugal","Champions","EuropaLeague","Conference"]

for league in leagueIDs:
    statsFile = league + "_All_TeamStats"
    reportFile = league + "_All_GameReports"
    pathXLS = "F:/ML_FootballPredictor/Final Data_New/" + league + "/XLS"
    
    entries = os.listdir(pathXLS)
    print(entries)
    
    teamStats = {}
    gameReports = {}
    
    teamStats = pd.DataFrame(teamStats)
    gameReports = pd.DataFrame(gameReports)
    
    
    for i in entries:
        print(i)
        if "_TeamStats.xlsx" in i:
            teamStats = teamStats.append(pd.read_excel(pathXLS + "/" + i))
            teamStats = teamStats.drop_duplicates()
    
        elif "_GameReports.xlsx" in i:
            gameReports = gameReports.append(pd.read_excel(pathXLS + "/" + i))
            gameReports = gameReports.drop_duplicates()
        #escrevendo em excelfile
        else:
            pass
    
    writer = pd.ExcelWriter(pathXLS + "\\" + statsFile + ".xlsx")
    teamStats.to_excel(writer,statsFile)
    writer.save()
    
    writer = pd.ExcelWriter(pathXLS + "\\" + reportFile + ".xlsx")
    gameReports.to_excel(writer,reportFile)
    writer.save()
