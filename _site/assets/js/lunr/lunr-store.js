var store = [{
        "title": "Common Quantitative Trading Strategies",
        "excerpt":"Quantitative trading strategies   Quantitative traders can employ a vast number of strategies, from the simple to the incredibly complex. Here are six common examples you might encounter:      Mean reversion   Trend following   Statistical arbitrage   Algorithmic pattern recognition   Behavioural bias recognition   ETF rule trading    ","categories": ["Finance"],
        "tags": ["Quant"],
        "url": "/finance/quant-strategies/",
        "teaser": null
      },{
        "title": "Timsort",
        "excerpt":"TimSort is a sorting algorithm based on Insertion Sort and Merge Sort. A stable sorting algorithm works in O(n Log n) time Used in Java’s Arrays.sort() as well as Python’s sorted() and sort(). First sort small pieces using Insertion Sort, then merges the pieces using merge of merge sort. You...","categories": ["Computer Science"],
        "tags": ["Algorithm"],
        "url": "/computer%20science/timsort/",
        "teaser": null
      },{
        "title": "How to add Latex to Jekyll",
        "excerpt":"Step 1. Set markdown engine to kramdown In your _config.yml change the engine to kramdown as follows # Build settings markdown: kramdown remote_theme: mmistakes/minimal-mistakes ... Step 2. Modify scripts.html We are now going to modify scripts.html and append the following content: &lt;script type=\"text/javascript\" async src=\"https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/latest.js?config=TeX-MML-AM_CHTML\"&gt; &lt;/script&gt; &lt;script type=\"text/x-mathjax-config\"&gt; MathJax.Hub.Config({ extensions:...","categories": ["Computer Science"],
        "tags": ["Jekyll"],
        "url": "/computer%20science/How-to-add-Latex-to-Jekyll/",
        "teaser": null
      },{
        "title": "Time Series Analysis 101",
        "excerpt":"These are all classical models of forecasting based on time series data, with slight differences. This post focuses on the conceptual differences to gain an intuition and overview. Autoregressive model Autoregression refers to a regression model based on itself (“auto”). Basically, it takes p number of time steps before the...","categories": ["Statistics"],
        "tags": ["Time Series Analysis"],
        "url": "/statistics/time_series_analysis_101/",
        "teaser": null
      },{
        "title": "Quantconnect Trending System",
        "excerpt":"Quantconnect is World’s Leading Algorithmic Trading Platform It provides a free algorithm backtesting tool and financial data so engineers can design algorithmic trading strategies. The following is a simple Trending System. It is for demo purpose only. The model uses Bollinger Bands(BB) and Relative Strength Index(RSI) Version 1 uses 4-hour...","categories": ["Finance"],
        "tags": ["Quant"],
        "url": "/finance/Quant_strategy/",
        "teaser": null
      },{
        "title": "Web Scraping 101",
        "excerpt":"This is a simple example of web scraping from wikipedia using python It requires two libraries: beautifulsoup4, requests import requests import sys import bs4 import re url = \"https://en.wikipedia.org/wiki/Jim_Simons_(mathematician)\" r = requests.get(url) if r.status_code == 200: html = r.text print(\"Success! Retrieved the page\") else: print(\"Error: downloading failed\") sys.exit(1) OUT[1]: Success!...","categories": ["Computer Science"],
        "tags": ["Web Scraping"],
        "url": "/computer%20science/web_scraping_101/",
        "teaser": null
      },{
        "title": "Star Wars API",
        "excerpt":"Star Wars API Jupyter Notebook HTML The following is the same content as above but reformatted: Intro Exploring the Star Wars API. This is an open and free-to-use API that provides all information you could possibly need about Star Wars. You can read about the project on the page https://swapi.dev/about...","categories": ["Computer Science"],
        "tags": ["API"],
        "url": "/computer%20science/star_wars_api/",
        "teaser": null
      },{
        "title": "Hidden Markov Model and Part-of-Speech Tagging",
        "excerpt":"We know that to model any problem using a Hidden Markov Model we need a set of observations and a set of possible states. The states in an HMM are hidden. In the part of speech tagging problem, the observations are the words themselves in the given sequence. As for...","categories": [],
        "tags": [],
        "url": "/projects/p1_hmm_tagger/",
        "teaser": "/assets/images/hmm.png"
      },{
        "title": "Mathematical Finance Notebook",
        "excerpt":"   Pricing option using Brownian Motion.   Notebook HTML   Notebook Github   Viewing jupyter notebook from github might show format error for equations.   ","categories": [],
        "tags": [],
        "url": "/projects/p2_math_finance/",
        "teaser": "/assets/images/stock_market.jpg"
      },{
        "title": "Information Retrieval using Term Frequency–Inverse Document Frequency",
        "excerpt":"I created a system for an Ad Hoc Information Retrieval task using TF-IDF weights and cosine similarity scores.                  My github project   Code  ","categories": [],
        "tags": [],
        "url": "/projects/p3_information_retrieval/",
        "teaser": "/assets/images/information_retrieval.png"
      },{
        "title": "Course Registration System",
        "excerpt":"   This is a java program that lets users perform course-related operations   The documentation for course registration system is below   Documentation   My github project  ","categories": [],
        "tags": [],
        "url": "/projects/p4_crs/",
        "teaser": "/assets/images/pan.png"
      },{
        "title": "Customer Service Scheduler",
        "excerpt":"   A java program that deals with customers service queue   My github project  ","categories": [],
        "tags": [],
        "url": "/projects/p5_css/",
        "teaser": "/assets/images/queue-teaser.png"
      },{
        "title": "A regular expression parser for dollars",
        "excerpt":"There are two python parsers in the project dollar_program.py 1 2 3 4 5 6 7 8 9 import sys,re regex = r\"(\\$?(?:(\\d+|a|half|quarter|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\\w+teen|\\w+ty|hundred|thousand|\\w+illion).)*((\\d+|and|((and|a)?.)?half( a)?|quarter|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\\w+teen|\\w+ty|hundred|thousand|\\w+illion))(\\s)?(dollar|cent)(s)?)|((\\$(?:\\d+.)*\\d+)(.(\\w+illion|thousand))?)\" with open(sys.argv[1], 'r') as f: test_str = f.read() matches = re.finditer(regex, test_str, re.MULTILINE) outFile=open(\"dollar_output.txt\",\"w\") for matchNum, match in enumerate(matches, start=1): outFile.write(match.group()+\"\\n\") outFile.close() telephone_regex.py 1 2 3...","categories": [],
        "tags": [],
        "url": "/projects/p6_regex/",
        "teaser": "/assets/images/regex.jpg"
      },{
        "title": "Sequence Labeling",
        "excerpt":"   The project is to create features for Maximum-entropy Markov model (MEMM) models to label the noun sequence.   My github project   Code  ","categories": [],
        "tags": [],
        "url": "/projects/p7_sequence_labeling/",
        "teaser": "/assets/images/sequence_labeling.png"
      },{
        "title": "Game of Life using OpenMP",
        "excerpt":"   This is n implementation of Conway’s Game of Life using C language, and parallel computing library OpenMP   My github project   My documentation   Code   External Resources:   https://playgameoflife.com/   http://battleship.ru/virus/conway.html  ","categories": [],
        "tags": [],
        "url": "/projects/p8_game_of_life/",
        "teaser": "/assets/images/gol.gif"
      },{
        "title": "Big Data Analytics",
        "excerpt":"   This is an analysis of Boston restaurants’ cleanliness data using our designed metrics from the data provided by Yelp and Boston government. The data manipulations for tables were done using Hive, and regression analyses were done using Spark.   My github project   My Medium writeup   My detailed report  ","categories": [],
        "tags": [],
        "url": "/projects/p9_big_data_analytics/",
        "teaser": "/assets/images/big_data_graph.jpg"
      }]
