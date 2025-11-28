import pandas as pd
import lyricsgenius
import os
from dotenv import load_dotenv
import rauth

exlude_terms = ["(Remix)", "(Live)", "(Demo)", "(Acoustic)", "(Instrumental)", "(Interlude)", 
                             "(Version)", "(Reference)", "interview", "(Explicit)", "(Clean)", "Reference)", 
                             "MTV", "Radio Edit", "Demo)", "Version)", "Reference", "[V2]", "[V3]", "[Remix]", "Remix)",
                             "mix"]

def getArtistSongs(artist, nb_songs, exclude_terms=exlude_terms):
    titles = []
    lyrics = []
    artist_genius = genius.search_artist(artist, max_songs = None, sort='title')
    songs = artist_genius.songs
    genius.excluded_terms = exclude_terms
    print(f"Fetched {len(songs)} songs for artist {artist_genius.name}")
    for song in songs:
        titles.append(song.title)
        lyrics.append(song.lyrics)
        if os.path.exists(f"lyrics/{artist}/{song.title}.txt"):
            print(f"Song {song.title} already exists. Skipping...")
            continue
        elif any(term.lower() in song.title.lower() for term in exclude_terms):
            print(f"Song {song.title} contains excluded terms. Skipping...")
            continue
        elif len(song.lyrics) < 100:
            print(f"Song {song.title} has insufficient lyrics. Skipping...")
            continue
        else:
            try:
                with open(f"lyrics/{artist}/{song.title}.txt", "w", encoding="utf-8") as f:
                    f.write(song.lyrics)
                    print(f"Saved lyrics for song {song.title}")
            except Exception as e:
                print(f"Error saving lyrics for song {song.title}: {e}")

#Get the key from the .env file and print
load_dotenv() 
key =  str(os.getenv("API_KEY"))
print("Got key from .env:", key)
client_access_token = os.getenv("API_KEY")


#creaete directories to store lyrics
artist_list = ["Kanye West"]
#["Kanye West", "Drake", "Jayz", "Kendrick Lamar", "Ice Cube", "Snoop Dogg", "Eminem", "Travis Scott", "Lil Wayne", "Dr. Dre", "Reference"]

nb_songs = 2000
genius = lyricsgenius.Genius(client_access_token, remove_section_headers=False, skip_non_songs=True, timeout=30, retries=3)
genius.skip_non_primary_artists = True
for artist in artist_list:
    os.makedirs(os.path.join('lyrics', artist), exist_ok=True)
    getArtistSongs(artist, nb_songs)

