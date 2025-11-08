import requests
from bs4 import BeautifulSoup
from pathlib import Path
import pandas as pd
pd.set_option('display.max_columns',None)

def extract_filepath_if_CSV_name(links):
    for link in links:
        if link[0]=='CSV':
          return [link]
    return links

def extract_links(html_path,suffix):
    response  = requests.get(html_path)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    links = [
        (a.get_text(strip=True), a['href']) for a in soup.find_all('a', href=True)
        if a['href'].endswith(suffix)
    ]
    links = extract_filepath_if_CSV_name(links)
    return links

def create_subfolders(subfolder_names, base_dir):
    for name in subfolder_names:
        folder = Path(base_dir) / name
        folder.mkdir(parents=True, exist_ok=True)

def extract_data_to_dir(name, href, html_base_path, base_dir):
    csv_path = html_base_path + '/' + href
    year = href.split('/')[1][:2] +'_' + href.split('/')[1][2:]
    save_path = base_dir / name / f'{year}.csv'
    print(f'Downloading {csv_path}')
    if save_path.exists():
        print(f'Skipping {save_path} as it already exists')
        return
    with requests.get(csv_path, stream=True) as r:
        r.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def extract_all_data_to_dir(html_path, base_dir, csv_links):
    subfolders = list(set([csv_link[0] for csv_link in csv_links]))
    create_subfolders(subfolders, base_dir)
    html_base_path = '/'.join(html_path.split('/')[:-1])
    for name, href in csv_links:
        try:
          extract_data_to_dir(name, href, html_base_path, base_dir)
        except Exception as e:
          print(f'Unable to download {name} for {href}:{e}')

def extract_one_csv_to_dir(html_path, base_dir, csv_links):
    name, href = csv_links[0]
    base_dir.mkdir(parents=True, exist_ok=True)
    save_path = base_dir / f'full.csv'
    csv_path = '/'.join(html_path.split('/')[:-1]) + '/' + href
    with requests.get(csv_path, stream=True) as r:
        r.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def extract_data_to_dir_from_html(html_path, base_dir):
    csv_links = extract_links(html_path,'.csv')
    if len(csv_links)==1:
        extract_one_csv_to_dir(html_path, base_dir, csv_links)
    else:
        extract_all_data_to_dir(html_path, base_dir, csv_links)



def move_full_csv_to_dir(csv_file, base_dir,columns='All', export=False, export_filepath=None):
    if columns=='All':
      df = pd.read_csv(csv_file)
    else:
      df = pd.read_csv(csv_file,usecols=columns)
    df = df.dropna(how='all')
    if export:
      df.to_csv(export_filepath, index=False)
    return df

def combine_data_from_dir(base_dir, columns = 'All', export=False, export_filepath = None):
    csv_files = list(base_dir.rglob('*.csv'))
    df_list = []
    if len(csv_files)==1:
      return move_full_csv_to_dir(csv_files[0], base_dir, columns=columns, export=export, export_filepath=export_filepath)
    for csv_file in csv_files:
        print(csv_file)
        try:
          if columns=='All':
            df = pd.read_csv(csv_file)
          else:
            df = pd.read_csv(csv_file,usecols=columns)
        except UnicodeDecodeError:
            # Fallback to Latin-1 if UTF-8 fails
            print(f"⚠️  Encoding issue in {csv_file.name}, retrying with Latin-1...")
            try:
              if columns == 'All':
                  df = pd.read_csv(csv_file, encoding='latin1')
              else:
                  df = pd.read_csv(csv_file, usecols=columns, encoding='latin1')
            except pd.errors.ParserError:
              print(f"⚠️  Parser issue in {csv_file.name}, retrying with python engine...")
              try:
                  df = pd.read_csv(csv_file, encoding='latin1', engine='python', on_bad_lines='skip')
              except Exception as e:
                  print(f"❌ Skipping {csv_file.name} due to parser error: {e}")
                  continue
        except pd.errors.ParserError:
            print(f"⚠️  Parser issue in {csv_file.name}, retrying with python engine...")
            try:
                df = pd.read_csv(csv_file, encoding='latin1', engine='python', on_bad_lines='skip')
            except Exception as e:
                print(f"❌ Skipping {csv_file.name} due to parser error: {e}")
                continue
        df = df.dropna(how='all')
        season = csv_file.stem
        div_name = csv_file.parent.stem
        df['Season'] = season
        df['Div_Name'] = div_name
        df_list.append(df)
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df = combined_df.dropna(axis=1, how='all')
    if export:
        combined_df.to_csv(export_filepath, index=False)
    return combined_df

def export_combined_csv_from_html(html_path, save_dir, combined_csv_filepath, columns='All'):
    extract_data_to_dir_from_html(html_path, save_dir)
    combined_df = combine_data_from_dir(save_dir, columns=columns, export=True, export_filepath=combined_csv_filepath)
    return combined_df

def extract_data_for_all_countries(base_dir):
    combined_dir = base_dir / 'Combined'
    combined_dir.mkdir(parents=True, exist_ok=True)
    links = [info for info in extract_links("https://www.football-data.co.uk/data.php",'.php') if 'Football Results' in info[0]]
    for link in links:
        name = link[0].split('Football Results')[0].strip()
        html_path = "https://www.football-data.co.uk/" + link[1]
        interim_dir = base_dir / name
        csv_filepath = combined_dir / f'{name}.csv'
        print(f'Processing {name} data')
        export_combined_csv_from_html(html_path, interim_dir, csv_filepath)


#save_dir = Path("/content/drive/MyDrive/Football_Data_3")
#extract_data_for_all_countries(save_dir)

#Above is useful for downloaded all data to folders
#Below for getting recent info and putting into df

country_divisions = {'England':['E0','E1','E2','E3','E4','EC']}


def get_upcoming_matches(countries,country_divisions):
    html_path = "https://www.football-data.co.uk/fixtures.csv"
    df = pd.read_csv(html_path)
    acceptable_divisions = [division for country in countries for division in country_divisions[country]]
    subset_df = df[df['Div'].isin(acceptable_divisions)]
    return subset_df


def get_recent_results_from_path(html_path, seasons):
    csv_links = extract_links(html_path,'.csv')
    df_list = []
    for div_name, href in csv_links:
        season = href.split('/')[1][:2] + '_'+ href.split('/')[1][2:]
        if season not in seasons:
            continue
        csv_path = '/'.join(html_path.split('/')[:-1]) + '/' + href
        try:
            df = pd.read_csv(csv_path)
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(csv_path, encoding='latin1')
            except pd.errors.ParserError:
                try:
                    df = pd.read_csv(csv_path, encoding='latin1', engine='python', on_bad_lines='skip')
                except Exception as e:
                    print(f"❌ Skipping {csv_path.name} due to parser error: {e}")
                    continue
        except pd.errors.ParserError:
            print(f"⚠️  Parser issue in {csv_path.name}, retrying with python engine...")
            try:
                df = pd.read_csv(csv_path, encoding='latin1', engine='python', on_bad_lines='skip')
            except Exception as e:
                print(f"❌ Skipping {csv_path.name} due to parser error: {e}")
                continue
        df = df.dropna(how='all')
        df['Div_Name'] = div_name
        df['Season'] = season
        df_list.append(df)
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df = combined_df.dropna(axis=1, how='all')
    return combined_df


def get_recent_results(countries, seasons):
    links = [info for info in extract_links("https://www.football-data.co.uk/data.php", '.php') if
             'Football Results' in info[0]]
    df_list = []
    for link in links:
        name = link[0].split('Football Results')[0].strip()
        if name not in countries:
            continue
        html_path = "https://www.football-data.co.uk/" + link[1]
        recent_df = get_recent_results_from_path(html_path, seasons)
        df_list.append(recent_df)
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df = combined_df.dropna(axis=1, how='all')
    return combined_df

#recent_df = get_recent_results(['England'],train_seasons)
#upcoming_df = get_upcoming_matches(['England'],country_divisions)
