"""Builds the documentation site published to GitHub Pages.

The site is generated from the Markdown that already lives in the repository, so
there is exactly one copy of every sentence. A published page that has drifted
from the README is worse than no published page, and the only reliable way to
prevent drift is to refuse to let the site have its own content.

The build is deliberately small and dependency-light: read Markdown, render it,
drop it into one template, rewrite the intra-repository links so they resolve on
a flat static host. No site generator, no theme to keep up to date, no plugin
that stops working on a runner three months from now.

Run it with ``python scripts/build_site.py --output _site``.
"""

from __future__ import annotations

import argparse
import html
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from markdown_it import MarkdownIt
from mdit_py_plugins.anchors import anchors_plugin
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.util import ClassNotFound

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

#: The published repository. Used for the "edit on GitHub" and source links.
REPOSITORY_URL: Final[str] = "https://github.com/kogunlowo123/rag-research-assistant"


@dataclass(frozen=True, slots=True)
class Page:
    """One Markdown source rendered to one HTML file."""

    source: Path
    slug: str
    title: str
    section: str

    @property
    def output_name(self) -> str:
        """The file name this page is written to."""
        return f"{self.slug}.html"


#: Every page on the site, in navigation order. Explicit rather than globbed:
#: the order of a navigation menu is an editorial decision, and a glob would
#: silently publish a file that was never meant to be public.
PAGES: Final[tuple[Page, ...]] = (
    Page(Path("README.md"), "index", "Overview", "Start here"),
    Page(Path("ARCHITECTURE.md"), "architecture", "Architecture", "Start here"),
    Page(Path("THREAT-MODEL.md"), "threat-model", "Threat model", "Start here"),
    Page(Path("docs/api.md"), "api", "HTTP API", "Reference"),
    Page(Path("docs/configuration.md"), "configuration", "Configuration", "Reference"),
    Page(Path("docs/evaluation.md"), "evaluation", "Evaluation", "Reference"),
    Page(Path("docs/operations.md"), "operations", "Operations", "Reference"),
    Page(Path("SECURITY.md"), "security", "Security policy", "Project"),
    Page(Path("CONTRIBUTING.md"), "contributing", "Contributing", "Project"),
    Page(Path("CHANGELOG.md"), "changelog", "Changelog", "Project"),
)

#: Maps a repository-relative Markdown path to the slug it is published under,
#: so a link written for GitHub still resolves on the site.
_SLUG_BY_SOURCE: Final[dict[str, str]] = {page.source.as_posix(): page.slug for page in PAGES}

#: Every link that is not absolute and not a bare fragment. All of them point at
#: something in the repository, and all of them have to be rewritten: the site
#: is flat, and a relative path that works on GitHub does not work here.
_RELATIVE_HREF: Final[re.Pattern[str]] = re.compile(
    r'href="(?!https?:|mailto:|#|/)([^"#]+)(#[^"]*)?"'
)

STYLESHEET: Final[str] = """\
:root {
  color-scheme: light dark;
  --bg: #ffffff;
  --fg: #1b1f24;
  --muted: #5b6572;
  --rule: #e2e6ea;
  --accent: #0b5fff;
  --code-bg: #f5f7f9;
  --sidebar-bg: #fafbfc;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0e1116;
    --fg: #e6edf3;
    --muted: #9aa7b4;
    --rule: #232c37;
    --accent: #6ea8ff;
    --code-bg: #161b22;
    --sidebar-bg: #11161d;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--fg);
  font: 16px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial,
    sans-serif;
}
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
.layout { display: flex; min-height: 100vh; align-items: flex-start; }
.sidebar {
  width: 260px;
  flex: 0 0 260px;
  background: var(--sidebar-bg);
  border-right: 1px solid var(--rule);
  padding: 28px 20px 48px;
  position: sticky;
  top: 0;
  max-height: 100vh;
  overflow-y: auto;
}
.sidebar h1 { font-size: 15px; letter-spacing: 0.02em; margin: 0 0 4px; }
.sidebar .tagline { color: var(--muted); font-size: 13px; margin: 0 0 22px; }
.sidebar h2 {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
  margin: 22px 0 8px;
}
.sidebar ul { list-style: none; margin: 0; padding: 0; }
.sidebar li { margin: 0 0 2px; }
.sidebar a {
  display: block;
  padding: 5px 10px;
  border-radius: 6px;
  color: var(--fg);
  font-size: 14px;
}
.sidebar a:hover { background: var(--rule); text-decoration: none; }
.sidebar a[aria-current="page"] { background: var(--accent); color: #fff; }
main { flex: 1 1 auto; min-width: 0; padding: 40px 44px 96px; max-width: 900px; }
main h1 { font-size: 30px; line-height: 1.25; margin: 0 0 20px; }
main h2 { font-size: 22px; margin: 36px 0 12px; padding-bottom: 6px;
  border-bottom: 1px solid var(--rule); }
main h3 { font-size: 17px; margin: 26px 0 8px; }
main img { max-width: 100%; }
code {
  background: var(--code-bg);
  padding: 0.15em 0.4em;
  border-radius: 4px;
  font-size: 0.88em;
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
}
pre {
  background: var(--code-bg);
  border: 1px solid var(--rule);
  border-radius: 8px;
  padding: 14px 16px;
  overflow-x: auto;
}
pre code { background: none; padding: 0; font-size: 13px; line-height: 1.55; }
table { border-collapse: collapse; width: 100%; display: block; overflow-x: auto; margin: 18px 0; }
th, td { border: 1px solid var(--rule); padding: 8px 12px; text-align: left; font-size: 14px; }
th { background: var(--code-bg); }
blockquote {
  margin: 18px 0;
  padding: 2px 16px;
  border-left: 3px solid var(--rule);
  color: var(--muted);
}
.header-anchor { color: var(--muted); text-decoration: none; opacity: 0; padding-left: 8px; }
h2:hover .header-anchor, h3:hover .header-anchor { opacity: 1; }
footer {
  margin-top: 56px;
  padding-top: 18px;
  border-top: 1px solid var(--rule);
  color: var(--muted);
  font-size: 13px;
}
@media (max-width: 860px) {
  .layout { display: block; }
  .sidebar { width: auto; flex: none; position: static; max-height: none;
    border-right: none; border-bottom: 1px solid var(--rule); }
  main { padding: 28px 20px 72px; }
}
"""

TEMPLATE: Final[str] = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{description}">
<link rel="stylesheet" href="style.css">
<link rel="stylesheet" href="pygments.css">
</head>
<body>
<div class="layout">
<nav class="sidebar">
<h1><a href="index.html">RAG Research Assistant</a></h1>
<p class="tagline">{tagline}</p>
{navigation}
<h2>Elsewhere</h2>
<ul><li><a href="{repository}">Source on GitHub</a></li></ul>
</nav>
<main>
{content}
<footer>
Generated from the Markdown in the repository.
<a href="{repository}/blob/main/{source}">Edit this page</a>.
</footer>
</main>
</div>
</body>
</html>
"""


def _highlight(code: str, language: str, _attrs: str) -> str:
    """Render a fenced code block with Pygments, falling back to plain text."""
    try:
        lexer = get_lexer_by_name(language) if language else guess_lexer(code)
    except ClassNotFound:
        return f"<pre><code>{html.escape(code)}</code></pre>"
    formatter = HtmlFormatter(nowrap=False, cssclass="highlight")
    return str(highlight(code, lexer, formatter))


def _renderer() -> MarkdownIt:
    """Build a CommonMark renderer with tables, strikethrough and heading anchors.

    Autolinking is deliberately off: the documentation writes its links
    explicitly, and linkify would turn every bare hostname in a configuration
    example into an anchor.
    """
    return (
        MarkdownIt("commonmark", {"highlight": _highlight})
        .enable(["table", "strikethrough"])
        .use(anchors_plugin, max_level=3, permalink=True, permalinkSymbol="#")
    )


def _resolve(target: str, *, relative_to: Path) -> str:
    """Resolve a Markdown link to a repository-relative path.

    Links are written for GitHub, where they resolve relative to the file that
    contains them. ``docs/configuration.md`` linking to ``operations.md`` means
    ``docs/operations.md``, and resolving it against the repository root instead
    would publish a link to a file that does not exist.
    """
    if target.startswith("/"):
        return target.lstrip("/")
    candidate = (relative_to / target).as_posix()
    parts: list[str] = []
    for part in candidate.split("/"):
        if part in {"", "."}:
            continue
        if part == ".." and parts:
            parts.pop()
        elif part != "..":
            parts.append(part)
    return "/".join(parts)


def _rewrite_links(rendered: str, *, source: Path) -> str:
    """Point Markdown links at their published HTML equivalents."""
    directory = source.parent

    def replace(match: re.Match[str]) -> str:
        target, fragment = match.group(1), match.group(2) or ""
        normalised = _resolve(target, relative_to=directory)
        slug = _SLUG_BY_SOURCE.get(normalised)
        if slug is not None:
            return f'href="{slug}.html{fragment}"'
        # Anything else in the repository — a source file, the licence, an
        # example directory — is linked on GitHub rather than becoming a 404.
        kind = "tree" if (REPO_ROOT / normalised).is_dir() else "blob"
        return f'href="{REPOSITORY_URL}/{kind}/main/{normalised}{fragment}"'

    return _RELATIVE_HREF.sub(replace, rendered)


def _navigation(current: Page) -> str:
    """Render the sidebar, marking the current page."""
    chunks: list[str] = []
    for section in dict.fromkeys(page.section for page in PAGES):
        items = [
            '<li><a href="{href}"{current}>{title}</a></li>'.format(
                href=page.output_name,
                current=' aria-current="page"' if page.slug == current.slug else "",
                title=html.escape(page.title),
            )
            for page in PAGES
            if page.section == section
        ]
        chunks.append(f"<h2>{html.escape(section)}</h2>\n<ul>{''.join(items)}</ul>")
    return "\n".join(chunks)


def _description(markdown: str, fallback: str) -> str:
    """Extract the first real paragraph of a document for the meta description."""
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "[!", "[![", ">", "|", "-")):
            continue
        return html.escape(stripped[:180], quote=True)
    return html.escape(fallback, quote=True)


_INTERNAL_HREF: Final[re.Pattern[str]] = re.compile(r'href="(?!https?:|mailto:|#)([^"]+)"')


def verify(output: Path) -> None:
    """Fail the build if any page links to something the site does not contain.

    A documentation site whose navigation 404s is worse than a README, so this
    runs as part of every build rather than as an optional extra step.
    """
    broken: list[str] = []
    for page in sorted(output.glob("*.html")):
        for match in _INTERNAL_HREF.finditer(page.read_text(encoding="utf-8")):
            target = match.group(1).split("#", 1)[0]
            if not target:
                continue
            if not (output / target).exists():
                broken.append(f"{page.name} -> {target}")
    if broken:
        message = "the generated site contains broken internal links: " + ", ".join(broken)
        raise RuntimeError(message)


def build(output: Path) -> int:
    """Render every page into ``output``. Returns the number of pages written."""
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    renderer = _renderer()
    (output / "style.css").write_text(STYLESHEET, encoding="utf-8")
    # types-Pygments leaves get_style_defs unannotated, so the call is untyped
    # in a strict context. It returns str; the ignore is scoped to that fact.
    style_defs: str = HtmlFormatter(cssclass="highlight").get_style_defs(  # type: ignore[no-untyped-call]
        ".highlight"
    )
    (output / "pygments.css").write_text(style_defs, encoding="utf-8")
    # Tells GitHub Pages not to run Jekyll over the output, which would
    # otherwise drop any file or directory whose name begins with an underscore.
    (output / ".nojekyll").write_text("", encoding="utf-8")

    tagline = "Hybrid retrieval, verified citations, and defence against injected instructions."
    written = 0
    for page in PAGES:
        source = REPO_ROOT / page.source
        if not source.is_file():
            message = f"documented page {page.source} is missing from the repository"
            raise FileNotFoundError(message)
        markdown = source.read_text(encoding="utf-8")
        body = _rewrite_links(renderer.render(markdown), source=page.source)
        (output / page.output_name).write_text(
            TEMPLATE.format(
                title=html.escape(
                    "RAG Research Assistant"
                    if page.slug == "index"
                    else f"{page.title} — RAG Research Assistant"
                ),
                description=_description(markdown, tagline),
                tagline=html.escape(tagline),
                navigation=_navigation(page),
                content=body,
                repository=REPOSITORY_URL,
                source=page.source.as_posix(),
            ),
            encoding="utf-8",
        )
        written += 1

    verify(output)
    return written


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Build the documentation site.")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "_site",
        help="Directory to write the site into. It is replaced if it exists.",
    )
    arguments = parser.parse_args(argv)
    count = build(arguments.output.resolve())
    sys.stdout.write(f"wrote {count} pages to {arguments.output}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
