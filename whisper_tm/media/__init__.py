"""Images & sounds: finding moments by what is seen or heard.

EmbeddingGemma 2 (Google, 10/2026) maps text, images, video and audio into
one vector space. Stills, clips, sound windows and spoken text of the
recordings are indexed (indexer.py → store.py) by a service that runs in an
environment of its own (service.py, environment.py, core/media_server.py);
a text, a hit or a file of one's own asks all of them (search.py), and the
language model adds what it measurably improves (llm.py). The search view's
tab and the settings page talk to api/media_search.py.
"""
