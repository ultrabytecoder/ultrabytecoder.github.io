---
layout: post
title: "Getting Started with Jekyll"
date: 2024-02-01 14:30:00 -0500
author: Your Name
tags: [jekyll, tutorial, github-pages]
description: "A quick guide to setting up Jekyll for GitHub Pages."
---

Jekyll is a static site generator that's perfect for blogs. Here's how to
get started.

## Installation

Install Jekyll via RubyGems:

```bash
gem install jekyll bundler
```

## Create a New Site

```bash
jekyll new my-blog
cd my-blog
bundle exec jekyll serve
```

## Write a Post

Create files in the `_posts/` directory using the naming convention:
`YYYY-MM-DD-title.md`.

## Deploy to GitHub Pages

1. Push your repository to GitHub
2. Go to **Settings → Pages**
3. Select **Deploy from branch**
4. Choose your branch (`main`) and `/ (root)` folder
5. Save — your site will be live at `https://username.github.io`

That's it! Happy blogging.
