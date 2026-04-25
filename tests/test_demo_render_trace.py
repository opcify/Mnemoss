"""Tests for ``demo/render_trace.py``.

Covers:
- Produces valid(ish) standalone HTML (DOCTYPE, head, body, player JS).
- Embedded trace JSON is parseable and matches the input.
- No external resource references (self-contained).
- Handles traces with and without breakdowns.
- CLI round-trip from a trace JSON file.
"""

from __future__ import annotations

import json
from pathlib import Path

from bench.backends.mnemoss_backend import MnemossBackend
from bench.backends.static_file_backend import StaticFileBackend
from demo.llm import StubLLM
from demo.render_trace import render_html
from demo.scenarios import SCENE1_PREFERENCE_RECALL
from demo.simulate import run_scenario
from demo.types import Trace
from mnemoss import FakeEmbedder


def _stub(scenario) -> StubLLM:
    return StubLLM([t.stub_response for t in scenario.turns])


# ─── structural HTML checks ───────────────────────────────────────


async def test_output_starts_with_doctype() -> None:
    async with StaticFileBackend() as be:
        trace = await run_scenario(SCENE1_PREFERENCE_RECALL, be, _stub(SCENE1_PREFERENCE_RECALL))
    html_out = render_html(trace)
    assert html_out.startswith("<!DOCTYPE html>")
    assert "</html>" in html_out


async def test_contains_required_markup_sections() -> None:
    async with StaticFileBackend() as be:
        trace = await run_scenario(SCENE1_PREFERENCE_RECALL, be, _stub(SCENE1_PREFERENCE_RECALL))
    html_out = render_html(trace, title="Scene 1 smoke")

    # Page chrome.
    assert "Scene 1 smoke" in html_out
    # Player controls.
    assert 'id="play"' in html_out
    assert 'id="restart"' in html_out
    assert 'data-speed="1"' in html_out
    # Stage panes.
    assert 'id="chat"' in html_out
    assert 'id="memory"' in html_out
    # Trace JSON script tag.
    assert 'id="trace-data"' in html_out
    # Embedded JS.
    assert "processEvent" in html_out


async def test_trace_json_roundtrip_via_embedded_script() -> None:
    """The trace JSON should be recoverable from the HTML body by
    locating the ``<script id="trace-data">`` block. This is the
    contract between the Python writer and the JS reader."""

    async with StaticFileBackend() as be:
        original = await run_scenario(SCENE1_PREFERENCE_RECALL, be, _stub(SCENE1_PREFERENCE_RECALL))
    html_out = render_html(original)

    needle_open = '<script id="trace-data" type="application/json">'
    start = html_out.index(needle_open) + len(needle_open)
    end = html_out.index("</script>", start)
    embedded = html_out[start:end]
    # Un-escape the </script> protection we applied at write time.
    embedded = embedded.replace("<\\/", "</")
    payload = json.loads(embedded)

    revived = Trace.from_dict(payload)
    assert revived.backend == original.backend
    assert len(revived.events) == len(original.events)


# ─── self-containment ────────────────────────────────────────────


async def test_no_external_network_references() -> None:
    """The player must be self-hostable. No CDN links, no <link
    rel="stylesheet">, no external fonts. Everything inlined."""

    async with StaticFileBackend() as be:
        trace = await run_scenario(SCENE1_PREFERENCE_RECALL, be, _stub(SCENE1_PREFERENCE_RECALL))
    html_out = render_html(trace)

    lowered = html_out.lower()
    forbidden = [
        '<link rel="stylesheet"',
        "cdnjs.",
        "unpkg.",
        "jsdelivr.",
        "googleapis.com/css",
        "<script src=",  # only inline <script>
    ]
    for term in forbidden:
        assert term not in lowered, f"external reference found: {term!r}"


async def test_closing_script_tag_in_trace_does_not_escape_the_script() -> None:
    """If a trace's content ever contains the literal ``</script>``, it
    must not break the embedded JSON block. The encoder escapes ``</``."""

    # Hand-build a trace with an adversarial content string.
    async with StaticFileBackend() as be:
        trace = await run_scenario(SCENE1_PREFERENCE_RECALL, be, _stub(SCENE1_PREFERENCE_RECALL))
    # Inject a problematic string into one event's content.
    trace.events[0].content = "A: </script><script>alert(1)</script>"

    html_out = render_html(trace)
    # The literal ``</script>`` must not appear before the end-of-block
    # ``</script>`` we actually wrote.
    first_close = html_out.find("</script>")
    # The first </script> should close the trace-data block — verify
    # it's preceded by id="trace-data".
    before = html_out[:first_close]
    assert 'id="trace-data"' in before
    # And the escaped form should appear in the JSON instead.
    assert "<\\/script>" in html_out


# ─── breakdown rendering path (Mnemoss traces only) ──────────────


async def test_mnemoss_trace_renders_breakdown_keys() -> None:
    """When breakdowns are present on hits, the JSON in the HTML must
    preserve the keys the JS player looks for."""

    async with MnemossBackend(embedding_model=FakeEmbedder(dim=16)) as be:
        trace = await run_scenario(SCENE1_PREFERENCE_RECALL, be, _stub(SCENE1_PREFERENCE_RECALL))
    html_out = render_html(trace)
    # The JS reads b.base_level, b.spreading, b.matching, b.noise.
    # Those string literals must appear in the HTML so the JS finds
    # them at runtime.
    assert '"base_level"' in html_out
    assert '"spreading"' in html_out
    assert '"matching"' in html_out
    assert '"noise"' in html_out


# ─── CLI ─────────────────────────────────────────────────────────


async def test_cli_reads_trace_and_writes_html(tmp_path: Path) -> None:
    """End-to-end: write a trace JSON, run the CLI, read the HTML back."""

    async with StaticFileBackend() as be:
        trace = await run_scenario(SCENE1_PREFERENCE_RECALL, be, _stub(SCENE1_PREFERENCE_RECALL))

    trace_json = tmp_path / "trace.json"
    trace_json.write_text(json.dumps(trace.to_dict()))
    out_html = tmp_path / "player.html"

    from demo.render_trace import main as cli_main

    rc = cli_main([str(trace_json), "--out", str(out_html), "--title", "CLI test"])
    assert rc == 0
    assert out_html.exists()
    body = out_html.read_text()
    assert "<!DOCTYPE html>" in body
    assert "CLI test" in body
