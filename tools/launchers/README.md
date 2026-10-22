# Advanced Windows launchers

For saved instrument replay, use **Start-Replay.cmd** at the project root.
The PowerShell helper here prepares the lightweight replay runtime automatically.

These other launchers are preserved for development and require a full `.venv`:

| Launcher | Purpose |
| --- | --- |
| `Start-Demo.cmd` | Start the original workflow demo and history service |
| `Stop-Demo.cmd` | Stop only the process managed by Start-Demo |
| `Run-Live.cmd` | Run a live generic instrument task (paid API calls) |
| `Run-RC1mx.ps1` | Run a bounded RC1mx + HFCal experiment (paid API calls) |

Launchers resolve the project root from their own location. Live runs additionally
require your local `.env` configuration. The duplicate `Start-Viewer.cmd` was
removed; the replay page's **Original logs** link opens the original history view.
