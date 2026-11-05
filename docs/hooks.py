"""MkDocs hooks (mkdocs.yml ``hooks``), applied to every page's Markdown before it is rendered; the files are unchanged.

- A relative link from a page to a repository file outside docs/ (``../README.md``, ``../ndm_gym/worldmodels/accel.py``)
  points at that file on GitHub, so the pages keep the links that work in the repository.
- ``[a] [b]`` outside code is two bracketed texts in these pages, not a reference link (which mkdocstrings' cross-reference
  lookup would try to resolve): the second bracket is escaped.
"""

import os
import re

LINK = re.compile(r"\]\((\.\./[^)\s#]+)(#[^)\s]*)?\)")
FENCE = re.compile(r"^\s*(```|~~~)")


def on_page_markdown(markdown, page, config, files):
    root = os.path.dirname(config["config_file_path"])
    docs = os.path.relpath(config["docs_dir"], root)

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
