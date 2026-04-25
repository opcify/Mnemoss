"""Live-recorded simulation assets for the Mnemoss launch.

The simulation is a scripted multi-turn conversation captured as a
deterministic JSON trace and replayed as a self-contained HTML
animation in the blog post. Zero LLM cost at replay, zero hosting
surface, zero abuse surface.

Top-level entry points:

- :func:`demo.simulate.run_scenario` — drive a Scenario against a
  MemoryBackend + LLM, capture trace events.
- :func:`demo.render_trace.render_html` — trace JSON → single-file
  HTML with an embedded player.

See ``demo/scenarios.py`` for the scripted scenes.
"""
