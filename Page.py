from pytube import Playlist
from pytube import YouTube
import re

playlist = Playlist("https://www.youtube.com/playlist?list=PLb02MaZXm5_Owrxlb_u_qWRX065eRM8wu")

playlist._video_regex = re.compile(r"\"url\":\"(/watch\?v=[\w-]*)")


print('Number of videos in playlist: %s' % len(playlist.video_urls))
for url in playlist.video_urls:
    print(url)

#yt = YouTube('https://www.youtube.com/watch?v=-tKVN2mAKRI')
#print(yt.title)
print()

for video in playlist.video_urls:
    YouTube(video).streams.first().download('/Users/khoi/Downloads')
#playlist.video_urls.download_all()


