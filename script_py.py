# Import Required Libraries
import requests
import numpy as np
import pandas as pd
import openpyxl
from bs4 import BeautifulSoup
import os
import re
from datetime import datetime
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# **PARTIE 1 - Scraping**
#Creating folders
BASE_DIR = "again enter path to your working directory here"
csv_folder_path_spain = os.path.join(BASE_DIR, "spain_csv")
csv_folder_path_sweden = os.path.join(BASE_DIR, "sweden_csv")

os.makedirs(csv_folder_path_spain, exist_ok=True)
os.makedirs(csv_folder_path_sweden, exist_ok=True)

## Spain
# URLs of Wikipedia pages for elections in Spain to scrape
elections_spain = {
             '2000' : 'https://en.wikipedia.org/wiki/Opinion_polling_for_the_2000_Spanish_general_election',
             '2004' : 'https://en.wikipedia.org/wiki/Opinion_polling_for_the_2004_Spanish_general_election',
             '2008' : 'https://en.wikipedia.org/wiki/Opinion_polling_for_the_2008_Spanish_general_election',
             '2011' : 'https://en.wikipedia.org/wiki/Opinion_polling_for_the_2011_Spanish_general_election',
             '2015' : 'https://en.wikipedia.org/wiki/Opinion_polling_for_the_2015_Spanish_general_election',
             '2016' : 'https://en.wikipedia.org/wiki/Opinion_polling_for_the_2016_Spanish_general_election',
             'April_2019' : 'https://en.wikipedia.org/wiki/Opinion_polling_for_the_April_2019_Spanish_general_election',
             'November_2019' : 'https://en.wikipedia.org/wiki/Opinion_polling_for_the_November_2019_Spanish_general_election',
             '2023' : 'https://en.wikipedia.org/wiki/Opinion_polling_for_the_2023_Spanish_general_election'
             }
# Function that handles cells that have an image instead of text
def get_cell_content(cell):
    img = cell.find('img')
    if img and img.get('alt'):
        return img.get('alt') # In the html code 'alt' is the name of the political party
    else:
        return cell.get_text(strip=True) #if condition not match it just get the text

#Scrapping function :
@st.cache_data
def spain_election_scraper(url, year):

    wiki_response = requests.get(url)
    wiki_soup = BeautifulSoup(wiki_response.text, 'html.parser')

    tables = wiki_soup.find_all('table', class_='wikitable')

    table = tables[0] #Selecting the first table of the wiki page

    header_row = table.find('tr')
    headers = []
    # Loops through each header cell and using our user defined function "get_cell_content"
    for cell in header_row.find_all(['th', 'td']):
        text = get_cell_content(cell)
        headers.append(text)

    # Select all rows except the header row
    data_rows = table.find_all('tr')[1:]
    data = []

    for row in data_rows:
        cols = row.find_all(['td', 'th'])
        row_data = []
        for cell in cols:

            text = get_cell_content(cell)
            colspan = int(cell.get('colspan', 1))
            row_data.extend([text] * colspan)
        #Keep only rows that have data and storing it
        if any(row_data): 
            data.append(row_data)


    df = pd.DataFrame(data)

    # If number of headers matches columns, set them as column names
    if headers:
        if len(headers) == df.shape[1]:
            df.columns = headers
        else:
            print("Error with headers")

    filename = f'spain_polls_{year}.csv'
    filepath = os.path.join(csv_folder_path_spain, filename)
    df.to_csv(filepath, index=False, encoding='utf-8-sig')

# Loop through the URLs for Spain
for year, url in elections_spain.items(): 
    spain_election_scraper(url, year)


## Sweden
# URLs of Wikipedia pages for elections in Sweden to scrape
elections_sweden = {
              '2014' : 'https://en.wikipedia.org/wiki/Opinion_polling_for_the_2014_Swedish_general_election',
              '2018' : 'https://en.wikipedia.org/wiki/Opinion_polling_for_the_2018_Swedish_general_election',
              '2022' : 'https://en.wikipedia.org/wiki/Opinion_polling_for_the_2022_Swedish_general_election'
              }
@st.cache_data             
def sweden_election_scraper(url, year):

    wiki_response = requests.get(url)
    wiki_soup = BeautifulSoup(wiki_response.text, 'html.parser')

    tables = wiki_soup.find_all('table', class_='wikitable')


    table = tables[0]

    header_row = table.find('tr')
    headers = []

    for cell in header_row.find_all(['th', 'td']):
        text = get_cell_content(cell)
        headers.append(text)

    data_rows = table.find_all('tr')[1:]
    data = []

    for row in data_rows:
        cols = row.find_all(['td', 'th'])
        row_data = []
        for cell in cols:
            text = get_cell_content(cell)
            colspan = int(cell.get('colspan', 1))
            row_data.extend([text] * colspan)

        if any(row_data):
            data.append(row_data)

    df = pd.DataFrame(data)

    if headers:
        if len(headers) == df.shape[1]:
            df.columns = headers
        else:
            print("Error with headers")

    filename = f'sweden_polls_{year}.csv'
    filepath = os.path.join(csv_folder_path_sweden, filename)
    df.to_csv(filepath, index=False, encoding='utf-8-sig')

# Loop through the URLs for Sweden   
for year, url in elections_sweden.items():
    sweden_election_scraper(url, year)


# **PARTIE 2 - Data Management**
def extract_year_from_filename(filename):
    match = re.search(r'(\d{4})', filename)
    return match.group(1) if match else None

# Function to standardize dates using regex
def parse_poll_date(date_str, default_year=None):
    patterns = [
        (r'(\d{1,2})\s*–\s*(\d{1,2})?\s*([A-Za-z]+)\s*(\d{4})', lambda m: (m.group(2) or m.group(1), m.group(3), m.group(4))),
        (r'(\d{1,2})\s*([A-Za-z]+)\s*(\d{4})', lambda m: (m.group(1), m.group(2), m.group(3))),
        (r'(\d{1,2})\s*–\s*(\d{1,2})?\s*([A-Za-z]+)', lambda m: (m.group(2) or m.group(1), m.group(3), default_year)),
        (r'(\d{1,2})\s*([A-Za-z]+)', lambda m: (m.group(1), m.group(2), default_year))
    ]
    for pattern, extractor in patterns:
        match = re.search(pattern, date_str)
        if match:
            day, month, year = extractor(match)
            try:
                return datetime.strptime(f"{day} {month} {year}", "%d %b %Y").strftime("%m/%d/%Y")
            except ValueError:
                return None
    return None  # Unrecognized format

## Spain
#Creating Folders
xlsx_folder_path_spain = os.path.join(BASE_DIR,'spain_xlsx')
os.makedirs(xlsx_folder_path_spain, exist_ok=True)

spain_political_parties_leaning = {
    "PP": "Centre-right",
    "PSOE": "Centre-left",
    "IU": "Left-wing",
    "CiU": "Centre-right",
    "PNV": "Centre-right",
    "CC": "Centre-right",
    "BNG": "Left-wing",
    "HB": "Far-left",
    "ERC": "Left-wing",
    "NI/IC": "Left-wing",
    "UPyD": "Centre",
    "IU-LV": "Left-wing",
    "IU-Upec": "Left-wing",
    "UPYD": "Centre",
    "ERC-CatSi": "Left-wing",
    "Compromís": "Left-wing",
    "C’s": "Centre-right",
    "Podemos": "Left-wing",
    "CDC": "Centre-right",
    "DiL": "Centre-right",
    "PACMA": "Left-wing",
    "Vox": "Far-right",
    "Sumar": "Left-wing",
    "Junts": "Centre-right",
    "EH Bildu": "Left-wing",
    "CUP": "Far-left",
    "UPN": "Centre-right",
    "EV": "Left-wing",
    "ERC-Sobiranistes": "Left-wing",
    "PDeCat": "Centre-right",
    "CCa": "Centre-right",
    "JxCat": "Centre-right",
    "NA+": "Centre-right",
    "CC-NCa": "Centre-left",
    "PRC": "Centre"
}
# Main function that process data from a CSV file
@st.cache_data
def process_spain_poll_data(csv_path, output_folder):

    filename = os.path.basename(csv_path)
    default_year = extract_year_from_filename(filename)

    #Load and clean the CSV file
    df = pd.read_csv(csv_path).rename(columns=str.strip)
    df.drop(columns=['Turnout', 'Lead'], errors='ignore', inplace=True)

    #Normalize dates
    df['Fieldwork date'] = pd.to_datetime(df['Fieldwork date'].apply(lambda x: parse_poll_date(x, default_year)))
    df['Sample size'] = pd.to_numeric(df['Sample size'].astype(str).str.replace(',', ''), errors='coerce')

    #Rename columns
    df.rename(columns={'Fieldwork date': 'poll_date', 'Polling firm/Commissioner': 'polling_organization', 'Sample size': 'sample_size'}, inplace=True)

    #Convert columns to numeric
    non_party_columns = ['poll_date', 'polling_organization', 'sample_size']
    for col in df.columns:
        if col not in non_party_columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.split("/").str[0], errors='coerce')

    #Convert data to long format(for mapping purpose)
    party_columns = [col for col in df.columns if col not in non_party_columns]
    # First row has final results
    election_results = df.iloc[0][party_columns].copy()  
    df = df.drop(index=0).reset_index(drop=True)

    df_long = df.melt(
        id_vars=non_party_columns, 
        value_vars=party_columns, 
        var_name='identity', 
        value_name='prediction_result')

    #Add final results and political leaning
    df_long['final_result'] = df_long['identity'].map(election_results)
    df_long['political_leaning'] = df_long['identity'].map(spain_political_parties_leaning)

    #Assign candidate numbers in the order they appear
    party_to_identifier = {}
    df_long['number'] = df_long['identity'].apply(lambda x: party_to_identifier.setdefault(x, f"candidate_{len(party_to_identifier) + 1}"))

    #Aggregate data to avoid duplicates
    df_long = df_long.groupby(['poll_date', 'sample_size', 'polling_organization', 'number', 'identity', 'final_result', 'political_leaning'], dropna=False)['prediction_result'].mean().reset_index()

    #Transform to wide format (Pivot Table)
    df_pivot = df_long.pivot(index=["poll_date", "sample_size", "polling_organization"], columns="number", values=["final_result", "prediction_result", "political_leaning", "identity"])

    #Sort columns to ensure correct order
    df_pivot.columns = [f"{var}_{candidate}" for var, candidate in df_pivot.columns]
    df_pivot = df_pivot[sorted(df_pivot.columns, key=lambda x: int(x.split('_')[-1]))]
    df_pivot.reset_index(inplace=True)

    #Remove rows where poll_date is missing
    df_pivot = df_pivot.dropna(subset=['poll_date'])

    #Export to Excel
    output_filepath = os.path.join(output_folder, os.path.splitext(filename)[0] + "_generalelection.xlsx")
    df_pivot.to_excel(output_filepath, index=False)

    print(f"File exported : {output_filepath}")

#Process every CSV file in the Spain CSV folder    
for file in os.listdir(csv_folder_path_spain):
    if file.endswith(".csv"):
        file_path = os.path.join(csv_folder_path_spain, file)
        process_spain_poll_data(file_path, xlsx_folder_path_spain)


## Sweden
#Creating Folders
xlsx_folder_path_sweden = os.path.join(BASE_DIR,'sweden_xlsx')
os.makedirs(xlsx_folder_path_sweden, exist_ok=True)

sweden_political_parties_leaning = {
    "S": "Centre-left",
    "M": "Centre-right",
    "Mp": "Left-wing",
    "MP": "Left-wing",
    "Fp": "Centre-right",
    "C": "Centre",
    "Sd": "Far-right",
    "SD": "Far-right",
    "V": "Left-wing",
    "L": "Centre-right",
    "Kd": "Centre-right",
    "KD": "Centre-right",
    "Fi": "Left-wing"
}
@st.cache_data
def process_sweden_poll_data(csv_path, output_folder):

    filename = os.path.basename(csv_path)
    default_year = extract_year_from_filename(filename)

    
    df = pd.read_csv(csv_path).rename(columns=str.strip)
    df.drop(columns=['Lead'], errors='ignore', inplace=True)

    #Handling possible different name
    if "Date" in df.columns:
        df.rename(columns={"Date": "Fieldwork date"}, inplace=True)

    #Make sure 'Fieldwork date' is present
    if "Fieldwork date" not in df.columns:
        raise ValueError(f"❌ 'Fieldwork date' column missing in {filename}")

    #Remove 'Samplesize' if present
    if "Samplesize" in df.columns:
        df.drop(columns=["Samplesize"], inplace=True)

    df['Fieldwork date'] = pd.to_datetime(df['Fieldwork date'].apply(lambda x: parse_poll_date(x, default_year)))

    #Rename columns
    df.rename(columns={'Fieldwork date': 'poll_date', 'Polling firm': 'polling_organization'}, inplace=True)
 
    non_party_columns = ['poll_date', 'polling_organization',]
    for col in df.columns:
        if col not in non_party_columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.split("/").str[0], errors='coerce')
  
    party_columns = [col for col in df.columns if col not in non_party_columns]
    election_results = df.iloc[0][party_columns].copy()  # Première ligne contient les résultats finaux
    df = df.drop(index=0).reset_index(drop=True)

    df_long = df.melt(id_vars=non_party_columns, value_vars=party_columns, var_name='identity', value_name='prediction_result')

    df_long['final_result'] = df_long['identity'].map(election_results)
    df_long['political_leaning'] = df_long['identity'].map(spain_political_parties_leaning)

    party_to_identifier = {}
    df_long['number'] = df_long['identity'].apply(lambda x: party_to_identifier.setdefault(x, f"candidate_{len(party_to_identifier) + 1}"))

    df_long = df_long.groupby(['poll_date', 'polling_organization', 'number', 'identity', 'final_result', 'political_leaning'], dropna=False)['prediction_result'].mean().reset_index()

    df_pivot = df_long.pivot(index=["poll_date", "polling_organization"], columns="number", values=["final_result", "prediction_result", "political_leaning", "identity"])

    df_pivot.columns = [f"{var}_{candidate}" for var, candidate in df_pivot.columns]
    df_pivot = df_pivot[sorted(df_pivot.columns, key=lambda x: int(x.split('_')[-1]))]
    df_pivot.reset_index(inplace=True)

    df_pivot = df_pivot.dropna(subset=['poll_date'])

    output_filepath = os.path.join(output_folder, os.path.splitext(filename)[0] + "_generalelection.xlsx")
    df_pivot.to_excel(output_filepath, index=False)

    print(f"File exported : {output_filepath}")

#Process every CSV file in the Sweden CSV folder
for file in os.listdir(csv_folder_path_sweden):
    if file.endswith(".csv"):
        file_path = os.path.join(csv_folder_path_sweden, file)
        process_sweden_poll_data(file_path, xlsx_folder_path_sweden)

        
# **PARTIE 3 - Data Visualization**
st.title(" Tableau interactif des pronostics de sondage pour les élections législatives en Espagne et en Suède")


#Path to the Excel files
dossier_data_spain = xlsx_folder_path_spain
dossier_data_sweden = xlsx_folder_path_sweden

#Country selection
pays = ['Espagne', 'Suède']
selection_pays = st.selectbox("Choix du pays", pays)

#Manage files for each country
df = None

if selection_pays == 'Espagne':
    annees = [2000, 2004, 2008, 2011, 2015, 2016, 2019, 2023]
    selection_annee = st.selectbox("Choisissez l'année d'élection législative", annees)

    if selection_annee == 2019:
        mois = st.selectbox("Choisissez le mois de l'élection", ['April', 'November'])
        fichier = os.path.join(xlsx_folder_path_spain, f'spain_polls_{mois}_2019_generalelection.xlsx')
    else:
        fichier = os.path.join(xlsx_folder_path_spain, f'spain_polls_{selection_annee}_generalelection.xlsx')

elif selection_pays == 'Suède':
    annees = [2014, 2018, 2022]
    selection_annee = st.selectbox("Choisissez l'année d'élection législative", annees)
    fichier = os.path.join(xlsx_folder_path_sweden, f'sweden_polls_{selection_annee}_generalelection.xlsx')

#Check if file exists and load data
try:
    df = pd.read_excel(fichier)
    st.write(f"Données chargées pour {selection_pays} en {selection_annee} :")

    #Display the DataFrame with st.data_editor
    st.write("Modifiez le tableau ci-dessous (vous pouvez ajouter des lignes) :")
    df_edited = st.data_editor(df)

    st.write("Tableau modifié :")
    st.write(df_edited)


except FileNotFoundError:
    st.error(f"File not found: {fichier}")

#If data is loaded, display candidate selection
if df is not None:
    num_candidats = [col.replace("prediction_result_candidate_", "") for col in df.columns if col.startswith("prediction_result_candidate_")]
    num_candidat_selectionne = st.selectbox("Choisissez un candidat", num_candidats)


    def graphique_sondage(df, num_candidat_selectionne):
        col_prediction = f'prediction_result_candidate_{num_candidat_selectionne}'
        col_final = f'final_result_candidate_{num_candidat_selectionne}'
        df[col_final] = df[col_final].fillna(df[col_final].mean())
        df_clean = df[['poll_date',col_final, col_prediction]].dropna()

        #Handle year 2014 in Sweden (filter date)
        if selection_pays == 'Suède' and selection_annee == 2014:
            date_debut = pd.Timestamp('2014-01-01')
            df_clean = df_clean[df_clean['poll_date'] >= date_debut]
        
        y_min = min(df_clean[col_prediction].min(),df_clean[col_final].min())
        y_max = max(df_clean[col_prediction].max(), df_clean[col_final].max())

        fig = make_subplots(specs=[[{'secondary_y':True}]])

        #Add poll points
        fig.add_trace(go.Scatter(
            x=df_clean["poll_date"],
            y=df_clean[col_prediction],
            mode='markers+lines',
            name=f'Sondage Candidat {num_candidat_selectionne}',
            marker=dict( size=5, color = df_clean[col_prediction], colorscale = 'Plasma', colorbar = dict(title = 'Valeur (%)')),
            line=dict(width=0.3 , color = 'grey')),
            secondary_y = False
        )

        fig.add_trace(go.Scatter(
            x=df_clean["poll_date"],
            y=df_clean[col_final],
            mode='lines',
            name=f'Résultat final pour le candidat{num_candidat_selectionne}',
            line=dict(width = 0.9 , color='#76FF7B')),
            secondary_y = True
        )

        #Trend line
        x_numeric = np.arange(len(df_clean))
        y = df_clean[col_prediction].values
        z = np.polyfit(x_numeric, y, 1)
        p = np.poly1d(z)

        
        fig.add_trace(go.Scatter(
            x=df_clean["poll_date"],
            y=p(x_numeric),
            mode='lines',
            name=f'Tendance Candidat {num_candidat_selectionne}',
            line=dict(dash='dot', color='#FF00FF',width = 2),
            showlegend=False
            ),
        secondary_y = False)

        #Layout settings
        fig.update_layout(
            title=f"Cartographie interactives des pronostics des sondages pour {selection_pays} en {selection_annee} pour le candidat {num_candidat_selectionne}",
            xaxis_title="Date du sondage",
            yaxis_title="Résultats du sondage (en %)",
            yaxis2_title="Résultat finaux de l'élection",
            xaxis=dict(showgrid=False, gridcolor = 'lightgray'),
            yaxis=dict(showgrid=True, gridcolor = 'lightgray', range = [y_min, y_max]),
            yaxis2=dict(showgrid=False, range=[y_min, y_max]),
            hovermode="x",
            template = 'plotly_white',
            showlegend = False
        )

        fig.update_yaxes(
            title_text="Résultat du sondage (en %), et tendance",
            title_font=dict(size=12, color='#FF00FF'),
            showgrid = True,
            gridcolor = 'lightgray',
            secondary_y= False
        )

        fig.update_yaxes(
            title_text="Résultat finaux de l'élection",
            title_font=dict(size=12,color='#76FF7B'),
            showgrid= False,
            secondary_y=True
        )


        return fig

    fig = graphique_sondage(df, num_candidat_selectionne)
    st.plotly_chart(fig, use_container_width=True)

if df is not None and not df.empty :
    col_prediction = f'prediction_result_candidate_{num_candidat_selectionne}'
    col_final = f'final_result_candidate_{num_candidat_selectionne}'

    if col_prediction in df.columns :
        moyenne = df[col_prediction].mean()
        mediane = df[col_prediction].median()
        minimum = df[col_prediction].min()
        maximum = df[col_prediction].max()
        resultat_final = df[col_final].iloc[-1]

        stats_df = pd.DataFrame({"Statistique":['Moyenne','Médiane','Minimum','Maximum','Résultat final'],
                                 'Valeur':[moyenne, mediane, minimum, maximum, resultat_final]
                                 })
        
        st.write("### Pronostics d'élections")
        st.dataframe(stats_df, hide_index=True)

        #Form to add a new poll
        st.write("### Ajouter un nouveau sondage")

        with st.form(key="new_poll"):
            new_poll_date = st.date_input("Date du sondage")
            new_polling_org = st.text_input("Nom de l'organisme de sondage")
            new_sample_size = st.number_input("Taille de l'échantillon", min_value=1)
    
            new_poll_results = {}
            for candidate in num_candidats:
                new_poll_results[candidate] = st.number_input(
                    f"Résultat du sondage pour candidat {candidate} (%)",
                      min_value=0.0, max_value=100.0, step=0.1)

            submit_button = st.form_submit_button(label="Ajouter le sondage")

        if submit_button:
            new_data = {
            "poll_date": pd.to_datetime(new_poll_date),
            "polling_organization": new_polling_org,
            "sample_size": new_sample_size,
            }

            for candidate, value in new_poll_results.items():
                new_data[f"prediction_result_candidate_{candidate}"] = value

            new_row = pd.DataFrame([new_data])
            df = pd.concat([df, new_row], ignore_index=True)
            df = df.sort_values(by="poll_date").reset_index(drop=True)

            #Save the updated data to Excel
            df.to_excel(fichier, index=False)

            st.success(f"Nouveau sondage ajouté avec succès ! Données mises à jour enregistrées dans : {fichier}")

    

    
    else:
        st.warning(f"La colonne {col_prediction} n'existe pas dans les données.")
else:
    st.error("Les données ne sont pas disponibles.")

st.markdown("---")
if st.button("❌ Quitter l'application"):
    os._exit(0)