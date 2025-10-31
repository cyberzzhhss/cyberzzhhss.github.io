---
title: "How to add Latex to Jekyll"
date: 2020-02-1
categories:
  - Computer Science
tags:
  - Jekyll
classes: wide

---

## Step 1. Set markdown engine to kramdown

In your `_config.yml` change the engine to kramdown as follows

```yml
# Build settings
markdown: kramdown
remote_theme: mmistakes/minimal-mistakes
...
```

## Step 2. Modify `scripts.html`

We are now going to modify `scripts.html` and **append** the following content:

```html
<script type="text/javascript" async
	src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/latest.js?config=TeX-MML-AM_CHTML">
</script>

<script type="text/x-mathjax-config">
   MathJax.Hub.Config({
     extensions: ["tex2jax.js"],
     jax: ["input/TeX", "output/HTML-CSS"],
     tex2jax: {
       inlineMath: [ ['$','$'], ["\\(","\\)"] ],
       displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
       processEscapes: true
     },
     "HTML-CSS": { availableFonts: ["TeX"] }
   });
</script>
```

## Step 3. That's it!

If you did everything properly then this should render nicely:

$$ e^{i \pi} = -1$$


<!--End mc_embed_signup-->
