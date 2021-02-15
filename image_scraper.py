from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
from webdriver_manager.chrome import ChromeDriverManager
from xlsxwriter import Workbook
import os
import json 
import requests # to sent GET requests
from bs4 import BeautifulSoup # to parse HTML
import cv2
import numpy as np
import urllib.request
import time


urls_df = pd.read_csv("linkler.csv", header=None )
urls_df.columns = ["category", "links"]
driver = webdriver.Chrome(ChromeDriverManager().install())
tuples =  tuple(zip(urls_df.category, urls_df.links))  

syc = 0 
j = 0
for category, url in tuples:
    print(len(tuples)-syc)
    driver.get('{}/videos?view=0&sort=p&flow=grid'.format(url))
    for _ in driver.find_elements_by_id("thumbnail"):
        i= 0 
        for img in driver.find_elements_by_css_selector("#img.style-scope.yt-img-shadow"):
            if (i==11):
                break

            a = str(img.get_attribute("src"))
            if a.find("https://i.ytimg.com") != -1:
                j+=1
                i+=1
                try:
                    req = urllib.request.Request(a+".png")
                    response = urllib.request.urlopen(req)
                    rr = response.read()
                    ba = bytearray(rr)
                    image = np.asarray(ba, dtype="uint8")
                    image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
                    cv2.imwrite("images/" +  category+'{:04d}'.format(j)+'{:04d}'.format(i) + ".png", image)
                    print("Saved " + category+'{:04d}'.format(j),'{:04d}'.format(i) + ".png")
                    print("1")
                except Exception as e:
                    print("Error Occured for Pokemon " + '{:04d}'.format(i))
                    print(str(e))
            
                
                print(a)


            else:
                continue
        break

        
#df.to_excel('output1.xlsx', engine='xlsxwriter')


