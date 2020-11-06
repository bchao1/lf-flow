import requests
from bs4 import BeautifulSoup

root = "https://www.turbosquid.com/Search/3D-Models/free/scene"

with requests.Session() as sess:
    res = sess.get(root)
    soup = BeautifulSoup(res.text, "html.parser")
    print(soup)