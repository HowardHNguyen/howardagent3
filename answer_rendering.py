"""Render model Markdown as formatted HTML without active links or remote media."""
from html import escape
from markdown_it import MarkdownIt


def answer_html(text):
    parser = MarkdownIt("commonmark", {"html": False, "linkify": False}).enable(["table", "strikethrough"])
    # Disallow remote-resource fetching and links derived from untrusted model text.
    parser.add_render_rule("image", lambda self, tokens, idx, options, env: escape(tokens[idx].content))
    parser.add_render_rule("link_open", lambda self, tokens, idx, options, env: "")
    parser.add_render_rule("link_close", lambda self, tokens, idx, options, env: "")
    body = parser.render(text)
    return '<div class="rag-answer">' + body + '</div>'


ANSWER_CSS = """<style>
.rag-answer {line-height: 1.6; overflow-wrap: anywhere;}
.rag-answer table {border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: .95rem;}
.rag-answer th, .rag-answer td {border: 1px solid #8885; padding: .55rem .7rem; text-align: left; vertical-align: top;}
.rag-answer th {background: #8882; font-weight: 600;}
.rag-answer tr:nth-child(even) {background: #8881;}
.rag-answer pre {white-space: pre-wrap; padding: .8rem; background: #8881;}
.rag-answer p {margin: .5rem 0 1rem;}
</style>"""
