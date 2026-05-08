from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from maa.agent.agent_server import AgentServer
from maa.tasker import Tasker

import play_song  # noqa: F401  Registers Maa custom actions.


def main() -> None:
    Tasker.set_log_dir(Path("./debug"))
    if len(sys.argv) < 2:
        print("Usage: python main.py <socket_id>")
        raise SystemExit(1)

    socket_id = sys.argv[-1]
    AgentServer.start_up(socket_id)
    AgentServer.join()
    AgentServer.shut_down()


if __name__ == "__main__":
    main()
