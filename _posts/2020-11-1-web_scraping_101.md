---
title: "Web Scraping 101"
excerpt_separator: <!--more-->
tags:
  - Web Scraping
categories:
  - Computer Science
classes: wide

---
This is a simple example of web scraping from wikipedia using python
It requires two libraries: beautifulsoup4, requests 


```python
import requests
import sys
import bs4
import re

url = "https://en.wikipedia.org/wiki/Jim_Simons_(mathematician)"
r = requests.get(url)

if r.status_code == 200:
    html = r.text
    print("Success! Retrieved the page")
else:
    print("Error: downloading failed")
    sys.exit(1)
```

    OUT[1]:
    Success! Retrieved the page

The following code extracts the occupations

```python
names = re.findall("\w+_",url) ## regex to filter names
names = names[0].split("_")
names = [i for i in names if i]  ## remove empty string
full_name = " ".join(names)    

soup = bs4.BeautifulSoup(html, 'html.parser')
role = soup.find(class_="role")
role_list = role.find_all('li')
for i in range(len(role_list)):
    role_list[i] = role_list[i].text
roles = ", ".join(role_list)

print("%s is %s." %(full_name, roles))
```

    OUT[2]:
    Jim Simons is mathematician, hedge fund manager, philanthropist.

