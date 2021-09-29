---
title: "Hidden Markov Model and Part-of-Speech Tagging"
# last_modified_at: 2020-10-11T08:55:27-04:00
last_modified_at: 2020-10-05
excerpt: "A python implementation of part-of-speech tagging using Hidden Markov Model"
header:
  # image: /assets/images/hmm.png
  teaser: /assets/images/hmm.png
# sidebar:
#   - title: "Role"
#     image: http://placehold.it/350x250
#     image_alt: "logo"
#     text: "Designer, Front-End Developer"
#   - title: "Responsibilities"
#     text: "Reuters try PR stupid commenters should isn't a business model"
# gallery:
#   - url: /assets/images/unsplash-gallery-image-1.jpg
#     image_path: assets/images/unsplash-gallery-image-1-th.jpg
#     alt: "placeholder image 1"
#   - url: /assets/images/unsplash-gallery-image-2.jpg
#     image_path: assets/images/unsplash-gallery-image-2-th.jpg
#     alt: "placeholder image 2"
#   - url: /assets/images/unsplash-gallery-image-3.jpg
#     image_path: assets/images/unsplash-gallery-image-3-th.jpg
#     alt: "placeholder image 3"


#https://www.freecodecamp.org/news/an-introduction-to-part-of-speech-tagging-and-the-hidden-markov-model-953d45338f24/?fbclid=IwAR3HGsgPWqFuK2Vwa1tnSonj_ZOYL-_BVNBXUI7qqNpzm2SvayxzlKARzdc

#https://www.freecodecamp.org/news/a-deep-dive-into-part-of-speech-tagging-using-viterbi-algorithm-17c8de32e8bc/

---
![hidden_markov_model2]({{site.url}}{{site.baseurl}}/assets/images/hmm2.svg)

We know that to model any problem using a Hidden Markov Model we need a set of observations and a set of possible states. The states in an HMM are hidden.

In the part of speech tagging problem, the observations are the words themselves in the given sequence.

As for the states, which are hidden, these would be the POS tags for the words.

![hidden_markov_model]({{site.url}}{{site.baseurl}}/assets/images/hmm.png)

The transition probabilities would be like <strong>P(VP | NP)</strong> which means the probability of the current word having a tag of Verb Phrase given that the previous tag was a Noun Phrase.

Emission probabilities would be <strong>P(Joe | NP)</strong>, which means the probability that the word is, say, Joe given that the tag is a Noun Phrase.


[My github project](https://github.com/cyberzzhhss/hmm_pos_tagger)

[Code](https://github.com/cyberzzhhss/hmm_pos_tagger/blob/main/hmm_pos_tagger.py)

