---
title: "Question Answering (SQuAD)"
excerpt: "A question answering bot that extract the answer from a paragraph."
last_modified_at: 2023-3-10
header:
  teaser: /assets/images/qa_bot.jpg
---


![image1]({{site.url}}{{site.baseurl}}/assets/images/qa_bot.jpg)


This task is about extractive question answering, where questions are posed about a document and answers are identified as spans of text within the document itself. 


- Conducted data pre-processing pipeline such as that included tokenization of questions and context, handling long contexts using stride, and mapping correct answer positions into tokenized sequences.

- Finetuned a pre-trained Transformer model, BERT, for a question-answering task on SQuAD (Stanford Question Answering Dataset) dataset, consisting of over 107,000 question-answer pairs.

- Evaluated BERT on the question-answering task from a given context with F1 score of 88.65% and an exact match score of 81.04%.

- Published the model on Hugging Face platform for interactive access.

[Demo at Hugging Face](https://huggingface.co/andrewshi/bert-finetuned-squad)
