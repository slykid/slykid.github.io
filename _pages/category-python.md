---
title: "Python"
layout: archive
permalink: /_post/python
---

{% assign posts = site.categories.python %}
{% for post in posts %} {% include archive-single.html type=page.entries_layout %} {% endfor %}