---
title: "A regular expression parser for dollars"
excerpt: "Regular expression python programs to parse the texts"
last_modified_at: 2020-09-21
header:
  teaser: /assets/images/regex.jpg
---
![information_retrieval]({{site.url}}{{site.baseurl}}/assets/images/regex.jpg)


There are two python parsers in the project

dollar_program.py
{% highlight python linenos %}
import sys,re 
regex = r"(\$?(?:(\d+|a|half|quarter|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\w+teen|\w+ty|hundred|thousand|\w+illion).)*((\d+|and|((and|a)?.)?half( a)?|quarter|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\w+teen|\w+ty|hundred|thousand|\w+illion))(\s)?(dollar|cent)(s)?)|((\$(?:\d+.)*\d+)(.(\w+illion|thousand))?)"
with open(sys.argv[1], 'r') as f:
    test_str = f.read()
matches = re.finditer(regex, test_str, re.MULTILINE)
outFile=open("dollar_output.txt","w")
for matchNum, match in enumerate(matches, start=1):
    outFile.write(match.group()+"\n")
outFile.close()
{% endhighlight %}

telephone_regex.py
{% highlight python linenos %}
import sys,re 
regex = r"[(]?\d{3}[)]?[(\s)?.-]\d{3}[\s.-]\d{4}"
with open(sys.argv[1], 'r') as f:
    test_str = f.read()
matches = re.finditer(regex, test_str, re.MULTILINE)
outFile=open("telephone_output.txt","w")
for matchNum, match in enumerate(matches, start=1):
    outFile.write(match.group()+"\n")
outFile.close()
{% endhighlight %}


This is the program file. It is possible to call the program on the command line with a text file as a parameter and output regexp matches in the format indicated below. For example,

    dollar_program.py target_text.txt  
    telephone_regex.py target_text.txt

dollar_output.txt -- this should contain the dollar amounts recognized by your program, one per line. The parts of the lines that are not part of the dollar amount should not be printed at all. 3 lines of example output might be something like this:\
$5 million\
$5.00\
five hundred dollars

telephone_output.txt – the output file for telephosne numbers,\
  e.g.,\
  212-345-1234\
  777-1000


[My github project](https://github.com/cyberzzhhss/regex_parser)