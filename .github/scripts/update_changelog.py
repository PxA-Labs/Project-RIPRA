#!/usr/bin/env python3
"""Append a changelog entry from a merged PR's title into CHANGELOG.md.

Environment variables:
  PR_NUM   - pull request number
  PR_TITLE - pull request title (conventional-commit style expected)
  PR_URL   - pull request HTML URL
"""

import os
import re

PR_NUM = os.environ["PR_NUM"]
PR_TITLE = os.environ["PR_TITLE"]
PR_URL = os.environ["PR_URL"]

# Map conventional-commit prefixes to Keep-a-Changelog sections
CATEGORY_MAP = {
    "feat": "Added",
    "fix": "Fixed",
    "docs": "Documentation",
    "refactor": "Changed",
    "test": "Testing",
    "perf": "Performance",
    "chore": "Maintenance",
    "ci": "CI/CD",
    "build": "Build",
    "revert": "Fixed",
}

prefix = PR_TITLE.split(":")[0] if ":" in PR_TITLE else ""
category = CATEGORY_MAP.get(prefix, "Other")

# Clean the title: "feat: add foo" -> "add foo"
title_clean = PR_TITLE.split(":", 1)[-1].strip() if ":" in PR_TITLE else PR_TITLE
entry = f"- {title_clean} ([#{PR_NUM}]({PR_URL}))\n"

with open("CHANGELOG.md", "r") as f:
    content = f.read()

HEADER = "## [Unreleased]"

if HEADER not in content:
    # No Unreleased section yet — create it after the top-level header
    content = content.replace(
        "# Changelog",
        f"# Changelog\n\n{HEADER}\n### {category}\n{entry}",
        1,
    )
else:
    hdr_idx = content.index(HEADER)
    after_hdr = content[hdr_idx + len(HEADER) :]

    cat_header = f"### {category}"
    if cat_header in after_hdr:
        # Insert right after the category header line
        cat_idx = after_hdr.index(cat_header)
        line_end = after_hdr.index("\n", cat_idx)
        pos = hdr_idx + len(HEADER) + line_end + 1
        content = content[:pos] + entry + content[pos:]
    else:
        # Find next top-level section (another ##) or EOF
        next_section = re.search(r"^## \[", after_hdr, re.MULTILINE)
        if next_section:
            pos = hdr_idx + len(HEADER) + next_section.start()
        else:
            pos = len(content)
        content = content[:pos] + f"\n### {category}\n{entry}" + content[pos:]

with open("CHANGELOG.md", "w") as f:
    f.write(content)
