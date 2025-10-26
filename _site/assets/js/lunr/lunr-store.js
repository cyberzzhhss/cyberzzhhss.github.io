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
        "excerpt":"Repost from Kasim Te These are all classical models of forecasting based on time series data, with slight differences. This post focuses on the conceptual differences to gain an intuition and overview. Autoregressive model Autoregression refers to a regression model based on itself (“auto”). Basically, it takes p number of...","categories": ["Statistics"],
        "tags": ["Time Series Analysis"],
        "url": "/statistics/time_series_analysis_101/",
        "teaser": null
      },{
        "title": "Quantconnect Trending System",
        "excerpt":"Quantconnect is World’s Leading Algorithmic Trading Platform It provides a free algorithm backtesting tool and financial data so engineers can design algorithmic trading strategies. The following is a simple Trending System. It is for demo purpose only. The model uses Bollinger Bands(BB) and Relative Strength Index(RSI) Version 1 uses 4-hour...","categories": ["Finance"],
        "tags": ["Quant"],
        "url": "/finance/quant_trending_system/",
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
        "title": "Santa Truth",
        "excerpt":"Growing up in the East, I have never understood the fascination that westerners have about Santa Claus. But I do know that it is culturally inappropriate to tell kids that Santa Claus doesn’t exist. Let us therefore indulge a thought experiment. Let us say a 5 year old kid, Kevin,...","categories": ["Philosophy"],
        "tags": ["Essay"],
        "url": "/philosophy/Santa_truth/",
        "teaser": null
      },{
        "title": "Curiosity & Perserverance",
        "excerpt":"This is a good tweet.   Riding my first #ebike today and it feels like the future has arrived. &mdash; Ben Cichy (@bencichy) November 22, 2019   ","categories": ["Philosohpy"],
        "tags": ["Quotes"],
        "url": "/philosohpy/perserverance/",
        "teaser": null
      },{
        "title": "Transformer Architecture Tutorial",
        "excerpt":"Here is the list of good resources to understand transformer architecture.           Distilled AI on Transformer            Harvard Annotated Transformer            Fast.ai Transformer Tutorial            Dive into Deep Learning on Transformer            The Little Book of Deep Learning            Understanding Deep Learning       ","categories": ["Computer Science","Data Science"],
        "tags": ["Artificial Intelligence","Machine Learning"],
        "url": "/computer%20science/data%20science/transformer_tutorial/",
        "teaser": null
      },{
        "title": "Energy-Based Models & Structured Prediction",
        "excerpt":"Energy-Based Models (EBMs) assign a scalar energy to configurations of variables and perform inference by minimizing energy. Intro We tackle structured prediction for text recognition: transcribing a word image into characters of variable length. We (1) build a synthetic dataset, (2) pretrain a sliding-window CNN on single characters, (3) align...","categories": ["Computer Science"],
        "tags": ["Energy-Based Models","Dynamic Programming","Viterbi","PyTorch"],
        "url": "/computer%20science/energy_based_model_character_recognition/",
        "teaser": null
      },{
        "title": "Question Answering (SQuAD)",
        "excerpt":"This task is about extractive question answering, where questions are posed about a document and answers are identified as spans of text within the document itself. Conducted data pre-processing pipeline such as that included tokenization of questions and context, handling long contexts using stride, and mapping correct answer positions into...","categories": [],
        "tags": [],
        "url": "/projects/p10_question_answering/",
        "teaser": "/assets/images/qa_bot.jpg"
      },{
        "title": "Handwritten Digits Recognition Web Application ",
        "excerpt":"Web application link Example 1 This project is a combination of machine learning and web app development. A project to create a digit recognition web application using Streamlit Developed a pipeline to train and validate a model on the MNIST Handwritten dataset. Fine-tuned the model until having a model that...","categories": [],
        "tags": [],
        "url": "/projects/p11_app4digits/",
        "teaser": "/assets/images/handwritten.png"
      },{
        "title": "Object Detection using Vicreg and RetinaNet",
        "excerpt":"Example 1 For this project we aim at carrying out an object detection task with variable sized input. We first researched on recent state-of-the-art methods, and then performed our downstream task using VICreg (Variance-Invariance-Covariance Regularization) (Bardes et al., 2021) to pretrain our ResNet backbone and RetinaNet (Lin et al., 2017)...","categories": [],
        "tags": [],
        "url": "/projects/p12_object_detection/",
        "teaser": "/assets/images/cv_dog.jpg"
      },{
        "title": "Photorealistic Style Transfer",
        "excerpt":"Example 1 We focused on a variant of Whitening and Coloring Transforms (WCT) called PhotoWCT, which specializes in photorealistic style transfer. We experimented with how training on different model architectures and layers affects the preservation of the original content structure in the stylized image. We used the Structural Similarity Index...","categories": [],
        "tags": [],
        "url": "/projects/p13_photorealistic_neural_style_transfer/",
        "teaser": "/assets/images/cv_nst2.png"
      },{
        "title": "Almgren Chriss Market Impact Model",
        "excerpt":"Liquidation Paths Price Impact Limit Order Book Realized Cost In this project, we worked with more than 100GB+ 3-month millisecond-level high-frequency NYSE trades and quotes tick data of more than 1000 tickers to calibrate the Almgren market impact model by applying nonlinear regression. We also formulated the Almgren-Chriss optimal execution...","categories": [],
        "tags": [],
        "url": "/projects/p14_almgren_market_impact_model/",
        "teaser": "/assets/images/trade_activity.png"
      },{
        "title": "Covariance Matrix Estimators Analysis",
        "excerpt":"Comparison of the optimal shrinkage function that maps the empirical eigenvalue $\\lambda_i$ onto ‘cleaned’ version, $\\xi_i$ Formula The question of building reliable estimators of covariance or of correlation matrices has a long history in finance, and more generally in multivariate statistical analysis. The performance of any mean-variance optimization scheme closely...","categories": [],
        "tags": [],
        "url": "/projects/p15_covariance_matrices_analysis/",
        "teaser": "/assets/images/cov_matrices.png"
      },{
        "title": "Big Data MovieLens",
        "excerpt":"Data Split Alternating Least Square Recommendation systems has been all around us. When we watch movies, listen to music, or order takeouts, we are all exposing our personal information, which allows companies to analyze our preferences and recommend items alike for better promotion and user engagement. In this project we...","categories": [],
        "tags": [],
        "url": "/projects/p16_big_data_movielens/",
        "teaser": "/assets/images/als.jpg"
      },{
        "title": "An amendament to bertology",
        "excerpt":"BERT architecture For this project we proposed a brand new method to study Bert models’ ability to utilize numeracy in several tasks, namely, classification and numeric-related question answering. RoBerta is a variant of the Bert model that was developed by Facebook AI. We compare roBerta model’s performance on the original...","categories": [],
        "tags": [],
        "url": "/projects/p17_bertology_amendament/",
        "teaser": "/assets/images/bert.png"
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
        "excerpt":"   This is an analysis of Boston restaurants’ cleanliness data using our designed metrics from the data provided by Yelp and Boston government. The data manipulations for tables were done using Hive, and regression analyses were done using Spark.   My github project   My detailed report  ","categories": [],
        "tags": [],
        "url": "/projects/p9_big_data_analytics/",
        "teaser": "/assets/images/big_data_graph.jpg"
      }]
