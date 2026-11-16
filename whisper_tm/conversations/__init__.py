"""Conversations: who talked with whom, and how — counted from transcripts.

Every transcript gets statistics of how its speakers talk: speech time,
words per sentence, speaker changes, talk shares, questions, the longest
monologue. They are counted from the segments alone (no model), stored
next to the transcript and kept current by every transcript write, so the
Conversations tab, Ask, Find and MCP clients can look for conversations by
who took part and how they talked — "an hour with Alice, Bob and Carol in
long sentences".

The parts:

* metrics.py — the counting: one transcript's segments in, its numbers out
  (pure, no database);
* store.py — the tables ``conv_stats``, ``conv_speakers`` and
  ``conv_windows``: written inside ``Database.upsert_transcript``, read by
  the background run and the queries;
* runner.py — the background run that counts what is missing or was
  counted by an older version of the rules (minutes of one CPU core for a
  large library);
* query.py — finding conversations: filters, people, baselines, kinds,
  and the stretch of a long recording where given people talk together.

Names are never stored with the numbers: a diarization label
(``SPEAKER_00``) means something only inside one recording, and who it is
can change any time in the voice library. They are looked up when a query
runs (rag/speakers.py: names typed in the transcript first, then the voice
map's majority), so renaming a person on the map shows at once.
"""
