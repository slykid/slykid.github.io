---
title: "윈도우 서버"
layout: archive
permalink: categories/server/windows
author_profile: true
sidebar_main: true
---


{% assign posts = site.categories.Server.Windows %}
{% for post in posts %} {% include archive-single2.html type=page.entries_layout %} {% endfor %}