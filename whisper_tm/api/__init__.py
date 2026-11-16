"""
API Blueprint Registration.
"""

from flask import Flask


def register_blueprints(app: Flask) -> None:
    """Register all API blueprints."""
    from whisper_tm.api.analysis import analysis_bp
    from whisper_tm.api.apidocs import apidocs_bp
    from whisper_tm.api.artifacts import artifacts_bp
    from whisper_tm.api.audio import audio_bp
    from whisper_tm.api.chat import chat_bp
    from whisper_tm.api.clips import clips_bp
    from whisper_tm.api.conversations import conversations_bp
    from whisper_tm.api.embedding import embedding_bp
    from whisper_tm.api.export import export_bp
    from whisper_tm.api.files import files_bp
    from whisper_tm.api.find import find_bp
    from whisper_tm.api.health import health_bp
    from whisper_tm.api.incidents import incidents_bp
    from whisper_tm.api.ingest import ingest_bp
    from whisper_tm.api.library import library_bp
    from whisper_tm.api.llm import llm_bp
    from whisper_tm.api.llm_jobs import llm_jobs_bp
    from whisper_tm.api.llm_queue import llm_queue_bp
    from whisper_tm.api.main import main_bp
    from whisper_tm.api.media import media_bp
    from whisper_tm.api.media_search import media_search_bp
    from whisper_tm.api.queue import queue_bp
    from whisper_tm.api.rag import rag_bp
    from whisper_tm.api.search import search_bp
    from whisper_tm.api.settings import settings_bp
    from whisper_tm.api.setup import setup_bp
    from whisper_tm.api.speaker import speaker_bp
    from whisper_tm.api.stats import stats_bp
    from whisper_tm.api.transcription import transcription_bp
    from whisper_tm.api.ui_prefs import ui_prefs_bp
    from whisper_tm.api.voices import voices_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(apidocs_bp)
    app.register_blueprint(files_bp)
    app.register_blueprint(export_bp)
    app.register_blueprint(clips_bp)
    app.register_blueprint(health_bp)
    app.register_blueprint(transcription_bp)
    app.register_blueprint(media_bp)
    app.register_blueprint(media_search_bp)
    app.register_blueprint(audio_bp)
    app.register_blueprint(find_bp)
    app.register_blueprint(queue_bp)
    app.register_blueprint(settings_bp)
    app.register_blueprint(setup_bp)
    app.register_blueprint(search_bp)
    app.register_blueprint(stats_bp)
    app.register_blueprint(conversations_bp)
    app.register_blueprint(speaker_bp)
    app.register_blueprint(llm_bp)
    app.register_blueprint(llm_jobs_bp)
    app.register_blueprint(llm_queue_bp)
    app.register_blueprint(analysis_bp)
    app.register_blueprint(ingest_bp)
    app.register_blueprint(embedding_bp)
    app.register_blueprint(rag_bp)
    app.register_blueprint(incidents_bp)
    app.register_blueprint(artifacts_bp)
    app.register_blueprint(voices_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(ui_prefs_bp)
    app.register_blueprint(library_bp)
