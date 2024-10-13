import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

consumer_key = "ZofARLhERbSms2FUfexYw4q4V" #Your API/Consumer key 
consumer_secret = "tcvAw8n1bnNDaKx5v9ZGlgDoEC1qVL2EaupSXHFvKISumKeHkQ" #Your API/Consumer Secret Key
access_token = "1790669755931635712-qkOlNtpbP8zlTHojvd4kvhfFsoyoiq"    #Your Access token key
access_token_secret = "LnGuXXTSJ7FwZiiuZZ7WLTLhd4S1HFsqeNIx3zlW0VpNy" #Your Access token Secret key

# def fetch_who_updates():
#     url = "https://www.who.int/emergencies/disease-outbreak-news"  # Adjust as needed
#     response = requests.get(url)
#     soup = BeautifulSoup(response.content, 'html.parser')
        
#     updates = soup.find_all('div', class_='sf-list-vertical')  # Adjust based on actual HTML structure
        
#     rwanda_marburg_updates = []
        
#     for update in updates:
#         title = update.find('p', class_='heading').text.strip()
#         date = update.find('p', class_='date').text.strip()
            
#         if 'marburg' in title.lower() and 'rwanda' in title.lower():
#             rwanda_marburg_updates.append({
#                 'title': title,
#                 'date': date
#             })
        
#     return pd.DataFrame(rwanda_marburg_updates)

# print(fetch_who_updates())