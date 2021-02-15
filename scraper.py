from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
from webdriver_manager.chrome import ChromeDriverManager
from xlsxwriter import Workbook
    
urls_df = pd.read_csv("linkler.csv", header=None )
urls_df.columns = ["category", "links"]
driver = webdriver.Chrome(ChromeDriverManager().install())
tuples =  tuple(zip(urls_df.category, urls_df.links))  
category_list = []
title_list = []
channel_name = []
views_list = []
years_list = []
link_list = []
photo_link_list =[]
syc = 0 
for category, url in tuples:
    print(len(tuples)-syc)
    try:
        driver.get('{}/videos?view=0&sort=p&flow=grid'.format(url))
        content = driver.page_source.encode('utf-8').strip()
        cname   = driver.find_element_by_css_selector("#text.style-scope.ytd-channel-name").text
        print(cname)
        soup = BeautifulSoup(content, 'lxml')
        titles = soup.findAll('a',id='video-title')
        views = soup.findAll('span',class_='style-scope ytd-grid-video-renderer')
        video_urls = soup.findAll('a',id='video-title')
        links   =  soup.findAll(id='thumbnail')
        print('Channel: {}'.format(url))
        print("links  =", links)
        i = 0 # views and time
        j = 0 # urls
        for title in titles[:10]:
            photo_link_list.append(links[j].get("src"))
            link_list.append("https://www.youtube.com"+video_urls[j].get('href'))
            views_list.append(views[i].text)
            years_list.append(views[i+1].text)
            category_list.append(category)
            title_list.append(title.text)
            channel_name.append(cname)
            #photo_link_list':photo_link_list}
            #print('\n{}\n{}\t{}\t{}\thttps://www.youtube.com{} , {}'.format(cname,title.text, views[i].text, views[i+1].text, video_urls[j].get('href'),links[j].get("src") ))
            i+=2
            j+=1
        syc+=1
        
    except:
        print(url , "\n bu url de sikinti oldu")
        syc+=1
        pass
    break
   
print("category = ", len(category_list),"channel_name = ", len(channel_name),"title_list = ", len(title_list),
    "views_list = ", len(views_list),"years_list = ", len(years_list),"link_list = ", len(link_list),
    "photo_link_list = ", len(photo_link_list))

df = pd.DataFrame({'category':category_list,'channel_name':channel_name,'title_list':title_list,
                   'views_list':views_list,'years_list':years_list,'link_list':link_list,
                   'photo_link_list':photo_link_list})

#df.to_excel('output1.xlsx', engine='xlsxwriter')
print(df)

