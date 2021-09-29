---
title: "Star Wars API"
excerpt_separator: <!--more-->
tags:
  - API
categories:
  - Computer Science
---
[Star Wars API Jupyter Notebook HTML](/assets/html/star_wars_api.html)

The following is the same content as above but reformatted:

<h3>Intro</h3>

Exploring the Star Wars API. This is an open and free-to-use API that provides all information you could possibly need about Star Wars.
You can read about the project on the page <code>https://swapi.dev/about</code> and access the technical documentation for the API on the page <code>https://swapi.dev/documentation </code>

Using an API requires that you get the relevant information from its documentation. This API has great documentation, so be sure to check it carefully. The documentation contains all the details you need to answer the questions below.

<h3>Example 1</h3>

You can access information about 10 planets in Star Wars by sending a get request (without any parameters) to

    http://swapi.dev/api/planets/


1.1) A for loop to print out the names of all 10 planets

```python
import requests, json
url = "http://swapi.dev/api/planets/"
response = requests.get(url)
data = json.loads(response.text)
for i in range(10):
    print(data['results'][i]['name'])
```

    OUT[1]:
    Tatooine
    Alderaan
    Yavin IV
    Hoth
    Dagobah
    Bespin
    Endor
    Naboo
    Coruscant
    Kamino


1.2) A function called get_planet_population that takes as an argument a string called 'planet_name'.

- the population of that planet as a number, if that planet is among the 10 planets listed in the data returned by an API call to 
<code>http://swapi.dev/api/planets/</code> and the API lists its population; or
- the special Python value None, if that planet is among the 10 planets listed in the data returned by an API call to <code>http://swapi.dev/api/planets/</code> and the API tells you the population is 'unknown'.
- the string "unknown planet", if that planet is not among the 10 planets listed in the data returned by an API call to <code>http://swapi.dev/api/planets/</code>.


```python
def get_planet_population(planet_name):
    import requests, json
    url = "http://swapi.dev/api/planets/"
    response = requests.get(url)
    data = json.loads(response.text)
    for i in range(10):
        if planet_name == data['results'][i]['name']:
            if data['results'][i]['population'] == 'unknown':
                return None
            return data['results'][i]['population']
    return "unknown planet"

print(get_planet_population("Tatooine")) ## vanilla exmaple planet
print(get_planet_population("Hoth")) ## planet with unknown population
print(get_planet_population("XYZ")) ## planet that doesn't exit
```

    OUT[2]:
    200000
    None
    unknown planet


1.3) Print the names of all planets, from among the 10 planets returned by a call to <code>http://swapi.dev/api/planets/</code>, that have a population less than or equal to 30000000 and whose climate description includes the word 'temperate'.

```python
import requests, json
url = "http://swapi.dev/api/planets/"
response = requests.get(url)
data = json.loads(response.text)
for i in range(10):
    if  data['results'][i]['population'] != 'unknown'  and int(data['results'][i]['population']) <= 30000000 and \
        'temperate' in data['results'][i]['climate']:
        print(data['results'][i]['name'])
```

    OUT[3]:
    Yavin IV
    Bespin
    Endor


<h3>Example 2</h3>

In this exmaple, I will use a while loop to issue requests for information about all starships in Star Wars. The API to use is located at
<code>http://swapi.dev/api/starships/</code>
Note that the data you get back is a dictionary that contains a key called 'next'. The value for that key is the URL to which you should send the next request using requests.get() to fetch the additional batch of information about the following 10 starships.



2.1) Retrieve information about all starships available via this API and store it in a list called 'starships'.

The typical way to fetch all results from an API is to use a while loop that will retrieve a batch of 10 results, add them to a list (or similar data structure) and then send another request for more results if the value for the key 'next' in the dictionary in the previous response contained a URL. When you retrieve the final batch of results and no more results are available, the server will send you a dictionary that will probably still contain results you need to add to the list but the value for key 'next' will be None (rather than a URL). Therefore, one common strategy is to have your while loop end when the value for key 'next' == None. Notice that None is a special value in Python (like True or False) and is not surrounded in quotes!
```python
starships = []
import requests, json
url = "http://swapi.dev/api/starships/"
number = 1
parameters = {'page': number}
response = requests.get(url, parameters)
data = json.loads(response.text)
while (data['next']!= None):
    response = requests.get(url, parameters)
    data = json.loads(response.text)
    length = len(data['results'])
    for i in range(length):
        starships.append(data['results'][i]['name'])
    number += 1
    parameters = {'page': number}
print(starships)
```

    OUT[4]:
    ['CR90 corvette', 'Star Destroyer', 'Sentinel-class landing craft', 'Death Star', 
    'Millennium Falcon', 'Y-wing', 'X-wing', 'TIE Advanced x1', 'Executor', 
    'Rebel transport', 'Slave 1', 'Imperial shuttle', 'EF76 Nebulon-B escort frigate', 
    'Calamari Cruiser', 'A-wing', 'B-wing', 'Republic Cruiser', 'Droid control ship', 
    'Naboo fighter', 'Naboo Royal Starship', 'Scimitar', 'J-type diplomatic barge', 
    'AA-9 Coruscant freighter', 'Jedi starfighter', 'H-type Nubian yacht', 
    'Republic Assault ship', 'Solar Sailer', 'Trade Federation cruiser', 
    'Theta-class T-2c shuttle', 'Republic attack cruiser', 'Naboo star skiff', 
    'Jedi Interceptor', 'arc-170', 'Banking clan frigte', 'Belbullab-22 starfighter', 'V-wing']


2.2) Print out the name of the fastest starship Star Wars. As indicated in the API documentation, speed is given by the MGLT (Maximum number of Megalights) attribute of a starship.

```python
fastest_starship = ''
max_speed = 0
import requests, json
url = "http://swapi.dev/api/starships/"
number = 1
parameters = {'page': number}
response = requests.get(url, parameters)
data = json.loads(response.text)
while (data['next']!= None):
    response = requests.get(url, parameters)
    data = json.loads(response.text)
    length = len(data['results'])
    for i in range(length):
        if data['results'][i]['MGLT'] != 'unknown' and int(data['results'][i]['MGLT']) > max_speed:
            max_speed = int(data['results'][i]['MGLT'])
            fastest_starship = data['results'][i]['name']
    number += 1
    parameters = {'page': number}
print("The fastest starship is %s and its MGLT(speed) is %d" %(fastest_starship, max_speed))
```

    OUT[5]:
    The fastest starship is A-wing and its MGLT(speed) is 120