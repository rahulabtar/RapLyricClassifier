import pandas as pd
import lyricsgenius
import os
from dotenv import load_dotenv
import rauth

# Load the .env file
load_dotenv()
rauth.OAuth2Session.get_access_token(
    client_id=os.getenv("CLIENT_ID"),
    client_secret=os.getenv("CLIENT_SECRET"),
    scope=["user-read-private", "user-read-email"],
    redirect_uri="http://localhost:8000/callback"
)
# Access the environment variables
client_access_token = os.getenv("API_KEY")
genius = lyricsgenius.Genius(client_access_token, remove_section_headers=True, skip_non_songs=True)
artist = "Kanye West"
nb_songs = 1
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