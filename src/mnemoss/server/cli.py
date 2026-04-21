"""``mnemoss-server`` CLI entry point.

Reads the bind address from env (``MNEMOSS_HOST``, ``MNEMOSS_PORT``)
and starts uvicorn with the factory so ``ServerConfig.from_env()`` runs
inside the worker process. No custom argument parsing — env vars are
the single source of truth for a server deployment.
"""

from __future__ import annotations

import os


def main() -> None:
    import uvicorn

    host = os.environ.get("MNEMOSS_HOST", "127.0.0.1")
    port = int(os.environ.get("MNEMOSS_PORT", "8000"))
    # factory=True makes uvicorn call create_app() itself, which in turn
    # calls ServerConfig.from_env() — so auth / storage root / embedder
    # selection all reflect the launcher's environment.
    uvicorn.run(
        "mnemoss.server.app:create_app",
        host=host,
        port=port,
        factory=True,
    )


if __name__ == "__main__":
    main()
