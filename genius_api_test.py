import pandas as pd
import lyricsgenius
import os
from dotenv import load_dotenv
import csv
import time

# Load the .env file
key =  str(os.getenv("API_KEY"))

# Access the environment variables
client_access_token = os.getenv("API_KEY")
client_access_token = ''
genius = lyricsgenius.Genius(client_access_token, remove_section_headers=True, skip_non_songs=True)
artist = "Kanye West"
nb_songs = 1
langage = "english"
titles = []
lyrics = []
artist_genius = genius.search_artist(artist, max_songs = nb_songs, sort='popularity')
songs = artist_genius.songs
song_number = 0
for song in songs:
    if song is not None:
        song_number += 1
        titles.append(song.title)
        lyrics.append(song.lyrics)
        print(f"Fetched song {song_number}: {song.title}")  

data = pd.DataFrame({'artist':artist, 'title':titles, 'lyrics':lyrics})
print(data)