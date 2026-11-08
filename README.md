# My Website

A personal blog and about page built with [Jekyll](https://jekyllrb.com/) and deployed on [GitHub Pages](https://pages.github.com/).

## Quick Start (Local)

```bash
# Install dependencies (requires Ruby + Bundler)
bundle install

# Serve the site locally
bundle exec jekyll serve

# Open http://localhost:4000
```

## Writing a New Blog Post

Create a new file in `_posts/` using the format:

```
_posts/YYYY-MM-DD-title-of-post.md
```

With front matter:

```markdown
---
layout: post
title: "Your Post Title"
date: 2024-03-15 10:00:00 -0500
author: Your Name
tags: [tag1, tag2]
description: "A short description for SEO."
---

Your content here in Markdown.
```

## Deploying to GitHub Pages

### Option A: User/Org Page (`username.github.io`)
1. Create a repository named `username.github.io`
2. Push this code to the `main` branch
3. Your site goes live at `https://username.github.io`

### Option B: Project Page (`username.github.io/repo-name`)
1. Create any repository
2. Push code to `main` branch
3. Go to **Settings → Pages → Source → Deploy from branch**
4. Select `main` and `/ (root)`
5. In `_config.yml`, set `baseurl: "/repo-name"`
6. Your site goes live at `https://username.github.io/repo-name`

## Customization

- **Site config:** Edit `_config.yml`
- **Colors & styles:** Edit `assets/css/style.css`
- **Social links:** Edit `_data/social.yml`
- **Navigation:** Edit `_includes/header.html`
- **Footer:** Edit `_includes/footer.html`

## License

MIT — feel free to use, modify, and share.
