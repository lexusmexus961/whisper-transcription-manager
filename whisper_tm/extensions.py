"""
Central extension registry.

Holds all singleton instances (DB, hardware, JobManager).
Initialized by app.py and imported by all other modules.
Breaks circular imports: app.py → api → extensions (NOT app).

IMPORTANT: Only import this module for access to global instances.
Never import directly from app.py (except in __main__.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from whisper_tm.database import Database
    from whisper_tm.hardware import HardwareProfile
    from whisper_tm.jobs import JobManager
    from whisper_tm.scanner.watcher import FileWatcher

# ── Singleton instances (set by app.create_app()) ─────────

db: Database | None = None
hw: HardwareProfile | None = None
job_manager: JobManager | None = None
file_watcher: FileWatcher | None = None
