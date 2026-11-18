"""MkDocs hooks (mkdocs.yml ``hooks``), applied to every page's Markdown before it is rendered; the files are unchanged.

- A relative link from a page to a repository file outside docs/ (``../README.md``, ``../ndm_gym/worldmodels/accel.py``)
  points at that file on GitHub, so the pages keep the links that work in the repository.
- ``[a] [b]`` outside code is two bracketed texts in these pages, not a reference link (which mkdocstrings' cross-reference
  lookup would try to resolve): the second bracket is escaped.
- A line ``{{ readme "Title" }}`` is replaced by the README's section ``## Title`` (``readme_section``), so that a page
  carries what the README carries (the leaderboard, the recommended settings) without a second copy to keep up to date.

``on_nav`` makes the sections' overview pages their sections' own pages (navigation.indexes) without renaming them to index.md.
"""

import os
import re

from mkdocs.structure.pages import Page

OVERVIEWS = {"tasks.md", "tasks/hoi.md", "tasks/muscles.md", "algorithms.md", "world_models.md"}
LINK = re.compile(r"\]\((\.\./[^)\s#]+)(#[^)\s]*)?\)")
FENCE = re.compile(r"^\s*(```|~~~)")
README = re.compile(r'^\{\{ readme "([^"\n]+)" \}\}$', re.M)
RELATIVE = re.compile(r"\]\((?!https?://|mailto:|#)([^)\s#]+)(#[^)\s]*)?\)")  # a link to a file of the repository
ANCHOR = re.compile(r"\]\((#[^)\s]*)\)")  # a link to another section of the README


class _OverviewPage(Page):
    is_index = True  # what the theme's navigation.indexes reads


def on_nav(nav, config, files):
    def walk(items):
        for item in items:
            if item.is_section:
                if item.children[0].is_page and item.children[0].file.src_uri in OVERVIEWS:
                    item.children[0].__class__ = _OverviewPage
                walk(item.children)

    walk(nav.items)
    return nav


def readme_section(title: str, root: str, page_dir: str, repo_url: str) -> str:
    """The README's section ``## title`` without its heading, for a page in ``page_dir``: a link to a file of the repository
    is made relative to the page (a page of the site stays one; any other file then goes to GitHub, as the pages' own
    ``../`` links do), and a link to another section of the README points at the README on GitHub. A missing section and
    HTML with a relative path (which would not be rewritten) are errors."""
    with open(os.path.join(root, "README.md")) as f:
        parts = re.split(r"^## (.+)\n", f.read(), flags=re.M)
    sections = dict(zip(parts[1::2], parts[2::2]))
    if title not in sections:
        raise KeyError(f"the README has no section '## {title}' (its sections: {sorted(sections)})")
    body = sections[title].strip("\n")
    if re.search(r'(src|href)="(?!https?://|#)', body):
        raise ValueError(f"the README's section '## {title}' holds HTML with a relative path, which a page of the site cannot include")
    body = RELATIVE.sub(lambda m: f"]({os.path.relpath(os.path.join(root, m.group(1)), page_dir)}{m.group(2) or ''})", body)
    return ANCHOR.sub(lambda m: f"]({repo_url}{m.group(1)})", body)


def on_page_markdown(markdown, page, config, files):
    root = os.path.dirname(config["config_file_path"])
    docs = os.path.relpath(config["docs_dir"], root)
    page_dir = os.path.join(root, docs, os.path.dirname(page.file.src_uri))
    markdown = README.sub(lambda m: readme_section(m.group(1), root, page_dir, config["repo_url"].rstrip("/")), markdown)

    def github(m):
        target = os.path.normpath(os.path.join(docs, os.path.dirname(page.file.src_uri), m.group(1)))
        if target == docs or target.startswith(docs + os.sep):
            return m.group(0)  # a page or file of the site: MkDocs resolves and checks it
        if not os.path.exists(os.path.join(root, target)):
            raise FileNotFoundError(f"{page.file.src_uri} links to {m.group(1)}, which is not in the repository")
        kind = "tree" if os.path.isdir(os.path.join(root, target)) else "blob"
        return f"]({config['repo_url'].rstrip('/')}/{kind}/main/{target}{m.group(2) or ''})"

    out, fenced = [], False
    for line in LINK.sub(github, markdown).split("\n"):
        if FENCE.match(line):
            fenced = not fenced
        elif not fenced and "] [" in line:  # outside inline code: the even parts between backticks
            line = "`".join(p.replace("] [", "] \\[") if i % 2 == 0 else p for i, p in enumerate(line.split("`")))
        out.append(line)
    return "\n".join(out)
