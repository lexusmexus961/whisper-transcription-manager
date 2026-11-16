"""
All default prompts for the LLM pipeline — ONE source.

config.default.yaml contains copies for user visibility (system prompt/presets
are user data and editable in settings); the task prompts defined here
(correction, translation, map/reduce) are intentionally NOT freely editable —
their format contracts (JSON schema, ID rules) must match the code.
"""

# ── Summarization ─────────────────────────────────────────────────────────
#
# A template is the complete system prompt of a summary: the dialog shows
# it as it is sent. All templates share one shape — a one-line task, what
# to produce, "Format:", "Rules:" — and end on the same two rules
# (only the transcript; its language), so they read alike and none of
# them forgets the guard against invented content.
#
# Timestamps reach the model as [H:MM:SS] (core/summarize.py:
# format_transcript), speakers as [H:MM:SS | name] — templates that work
# with them say "timestamps" / "speaker labels" in those words; the dialog
# uses that to point out when the matching source option is off.

SUMMARY_DEFAULT = """\
Summarize the transcript as a structured overview.

Open with two or three sentences: what kind of recording this is, who is
talking and what comes out of it. Then cover the content in short sections
with headings (## …), in the order it comes up.

Be concrete: names, numbers, decisions, results and open questions belong
in it, not just the topics ("the budget rises to €12,000", not "the budget
is discussed").

Rules:
- Only what is said in the transcript — no outside knowledge, no guesses.
- If something stays unclear or is cut off, say so instead of smoothing it
  over.
- No preamble such as "Here is the summary" — start with the content.
- Write in the language of the transcript."""


# ── Built-in templates (the "Template" list of the summarize dialog) ──────
# ONE source: config.default.yaml mirrors them for new installations; an
# existing config fetches new versions via Settings → AI tasks → "⬇ Built-in
# templates". Templates the user wrote are never touched.

STANDARD_PRESETS: dict[str, str] = {
    "Summary": SUMMARY_DEFAULT,
    "Short Summary": """\
Summarize the transcript in at most five sentences of running text — no
headings, no lists.

Say what it is about, who is talking and what comes out of it.

Rules:
- Only what is said in the transcript — no outside knowledge, no guesses.
- No preamble such as "In this video…" — start with the content.
- Write in the language of the transcript.""",
    "Bullet Points": """\
List the most important statements of the transcript as bullet points.

Format: one line per point, starting with "- ", the most important first,
at most 20 points.

Rules:
- Every point stands on its own and is concrete — with the names, numbers
  and results ("the price rises to €12", not "prices are discussed").
- No repetitions, no topic headings without content.
- Only what is said in the transcript — no outside knowledge, no guesses.
- Write in the language of the transcript.""",
    "Meeting Minutes": """\
Write minutes of the conversation.

Format — leave out any section there is nothing for:
## Participants
## Topics
## Decisions
## Open questions
## Tasks
One line per task: who · what · by when (as far as it was said).

Rules:
- Record results and decisions, not the course of the conversation.
- Participants: the names used in the conversation; where there are none,
  the speaker labels.
- Mark anything left unresolved as "(open)" instead of interpreting it.
- Only what is said in the transcript — no outside knowledge, no guesses.
- Write in the language of the transcript.""",
    "Chapter Marks": """\
Divide the transcript into chapters, like the chapter list under a video.

Format: one line per chapter — the timestamp, then a title:
[0:00:00] Title

Rules:
- The first chapter starts at [0:00:00]; every further one at the
  timestamp of the line where its topic begins, copied exactly — never
  estimated or rounded.
- A new chapter only where the topic really changes, not with every
  sentence.
- Titles of at most six words that name the content concretely.
- If the transcript has no timestamps, say so in one sentence and list the
  topics instead.
- Write in the language of the transcript.""",
    "Timeline": """\
Write a chronological timeline of what happens in the transcript.

Format: one entry per line — the timestamp in bold, then what happens:
**[0:12:34]** — what happens

Rules:
- Only moments that move things forward; group related lines into one
  entry.
- Copy each timestamp exactly from the transcript line — never estimate.
- If the transcript has no timestamps, say so in one sentence and keep the
  order without times.
- Only what is said in the transcript — no outside knowledge, no guesses.
- Write in the language of the transcript.""",
    "Facts & Figures": """\
Collect every checkable fact in the transcript: numbers, amounts, dates,
names, places, products, companies, measurements.

Format: one line per fact:
- **Fact** — the context in which it was mentioned

Rules:
- Only what was literally said — don't convert, round or complete
  anything.
- Mark uncertain or contradictory statements with "(uncertain)".
- If there are no such facts, say so in one sentence.
- Write in the language of the transcript.""",
    "Speaker Analysis": """\
Describe each speaker in the transcript.

Format — per speaker, their name or speaker label as a heading (## …):
- **Role and stance** — one sentence
- **Main topics and key statements**
- **Notable** — turning points, contradictions, open questions

Rules:
- Base every statement on specific passages; don't characterize anyone
  beyond what they say.
- If the transcript has no speaker labels, say so and describe the voices
  only as far as the content allows.
- If a speaker label seems to mix up two people, point it out.
- Write in the language of the transcript.""",
    "Questions & Answers": """\
Extract the questions asked in the transcript, each with its answer.

Format:
**Q:** the question, shortened to its core
**A:** the answer

Rules:
- Keep unanswered questions, with "**A:** (open)".
- Leave out rhetorical questions and back-channel ones like "What?".
- In the order of the transcript.
- Only what is said in the transcript — no outside knowledge, no guesses.
- Write in the language of the transcript.""",
    "Highlights": """\
Pick the most memorable moments of the transcript — funny, surprising,
emotional or decisive ones.

Format per moment:
### [0:12:34] Short title
**Who:** the people involved
**What happens:** one or two sentences
**Quote:** "the line that carries the moment", verbatim

Rules:
- At most 10 moments, in chronological order — quality over quantity.
- Copy each timestamp exactly from the line where the moment begins.
- Quotes verbatim from the transcript, never paraphrased or invented.
- Leave out routine passages; if nothing stands out, say so.
- Write in the language of the transcript.""",
}


# ── Map-reduce: transcripts too long for one call (core/summarize.py) ─────
# The map step condenses one excerpt at a time into notes; the reduce step
# runs the template over all notes as if they were the transcript. The map
# step sees the template as a TASK TO COLLECT FOR, not to carry out:
# carried out per excerpt, its format would multiply ("at most 20 points"
# × 8 excerpts), and a neutral collection lost exactly what special
# templates look for (a joke is not "substantively relevant").

MAP_SYSTEM = """\
You condense one excerpt of a long transcript into notes. The notes of all
excerpts are combined later to carry out a task, so keep everything that
task could need.

Capture what is relevant in the excerpt: topics, statements, arguments,
decisions, questions and results, with names, numbers, dates, places and
technical terms exactly as they appear. Keep the timestamp of each point
([0:12:34]) and short verbatim quotes of striking lines. Dense notes, not
polished prose.

Don't introduce ("In this excerpt…"), evaluate or interpret, and add
nothing that isn't in the excerpt.

Write in the language of the transcript."""

MAP_TASK_TEMPLATE = """\
The task the notes are for — don't carry it out, collect for it:
<task>
{task}
</task>"""


def map_system(task: str = "") -> str:
    """The map step's system prompt, pointed at the task it collects for."""
    task = (task or "").strip()
    if not task:
        return MAP_SYSTEM
    return MAP_SYSTEM + "\n\n" + MAP_TASK_TEMPLATE.format(task=task)


MAP_USER_TEMPLATE = """\
Transcript excerpt {number}/{total} ({from_value}–{to_value}):

{text}

Notes for this excerpt:"""

# More notes than fit into the reduce call: pairs are merged first
CONDENSE_USER = """\
Merge these notes into ONE set of dense notes — keep the timestamps and
the quotes:

"""

REDUCE_INTRO_TEMPLATE = """\
The following {count} material collections cover one long transcript
completely, in chronological order. Treat them as the transcript itself
and carry out your task on them.

- Merge what appears in several collections instead of repeating it.
- The timestamps in the notes come from the transcript — use them where
  your task asks for times. The ranges in the part headings only show
  where each part lies.
- Don't mention that the material came in parts.

{parts}"""

# ── Correction ────────────────────────────────────────────────────────────
# Two-part: the RULES (what to correct, what to leave alone) are user data
# and replaceable via settings (llm.correction.system_prompt). The FORMAT
# (marker syntax, JSON output, example) is the contract with the validating
# code and ALWAYS appended — even to a user prompt.

CORRECTION_RULES = """\
You check auto-generated transcripts (Whisper ASR) for recognition errors.
You are a proofreader, not an editor.

CORRECT ONLY
- Words that make no sense in sentence context and where a phonetically
  similar alternative is obviously meant
- garbled proper names when the same person or thing is spelled correctly
  elsewhere in the context
- clear homophone confusions ("more"/"sea", "again"/"against")

LEAVE ALONE
- spoken language: filler words, repetitions, incomplete sentences, dialect,
  colloquial speech — this is correctly transcribed
- punctuation, capitalization, grammar
- anything plausible, even if marked: low confidence is a hint, not proof
- segments whose meaning you can't reliably reconstruct without the audio"""

CORRECTION_FORMAT = """\
INPUT
Numbered segments. ⟨text|score⟩ marks a passage where speech
recognition was uncertain (0 = very uncertain, 1 = certain). ⚠ after
the number means: the whole segment is uncertain. Lines under "CONTEXT"
and lines with a parenthesized (id) are only for understanding and must
NOT be changed.

OUTPUT
Two fields:

"edits" — only the segments you REALLY change. For each: the
full corrected sentence without ⟨⟩ markers. The new text MUST
differ from the original.
  "reason": in the language of the transcript, at most eight words.
  "confidence": 0–1, how sure you are about this correction.

"checked_ok" — the IDs of the segments you checked and
consider fine. ONLY the numbers, no text. This is the short way to
say "looked at it, no error" and is explicitly welcome.

EXAMPLE
Input:
  [12] We'll meet on ⟨Wendsday|0.22⟩ at ten.
  [13]⚠ and then uh I'm not really sure
  [14] The ⟨see|0.31⟩ money we don't need.
Correct output:
  {"edits": [
    {"id": 12, "text": "We'll meet on Wednesday at ten.",
     "reason": "Typo in the weekday", "confidence": 0.95},
    {"id": 14, "text": "The sea money we don't need.",
     "reason": "Homophone: sea instead of see", "confidence": 0.8}
  ], "checked_ok": [13]}

If you find nothing to change, {"edits": [], "checked_ok": [ … ]} is the
correct answer. Never invent corrections just to deliver something —
if you're unsure, the ID belongs under "checked_ok"."""


def correction_system(own_rules: str = "") -> str:
    """The complete correction system prompt.

    ``own_rules`` (own rules, from llm.correction.system_prompt) replaces ONLY the
    rules part; the format contract is always appended after it — otherwise
    a user prompt could bypass the JSON schema and validation
    would silently discard everything.
    """
    rules = (own_rules or "").strip() or CORRECTION_RULES
    return rules + "\n\n" + CORRECTION_FORMAT


# Combined default (rules + format) — for token estimates
# and tests that check the overall contract.
CORRECTION_SYSTEM = correction_system()

CORRECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "edits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "text": {"type": "string"},
                    "reason": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["id", "text", "reason", "confidence"],
                "additionalProperties": False,
            },
        },
        # Plain ID list instead of "every segment again in full text".
        # The model WANTS to report what it checked (most returns are
        # unchanged confirmations) — with
        # a dedicated field for that it costs one number instead of a
        # whole sentence. With full-text confirmations, a single
        # chunk could take many minutes because the model retyped every segment.
        "checked_ok": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["edits", "checked_ok"],
    "additionalProperties": False,
}

# ── Translation ───────────────────────────────────────────────────────────

TRANSLATION_SYSTEM_TEMPLATE = """\
You translate subtitles into {target_language}.

THE SEGMENTATION IS FIXED
Each numbered line is its own subtitle with a fixed display time.
Translate each line individually and return exactly one translation for
EVERY ID. Never merge, split, reorder, or
omit lines. A segment can end mid-sentence — translate so that
the parts connect in order.

LANGUAGE AND TONE
- Natural, spoken subtitle language; keep the length roughly like the
  original (subtitles must stay readable).
- Preserve register and tone: casual stays casual, formal stays
  formal, curses stay curses.
- Keep proper names, titles, and technical terms consistent — go by
  the supplied context of already translated lines.
- Replace interjections and filler words with the usual equivalents of
  the target language, don't transliterate sound for sound.
- If the original repeats a phrase multiple times, repeat it just as often.

ONLY THE TRANSLATION
No explanations, no parenthetical remarks, no notes about the original,
no alternative suggestions. If a line is incomprehensible,
translate it as literally as possible.

EXAMPLE (French → English)
Input:
  [40] Alors j'ai dit qu'on devait
  [41] absolument le revoir encore une fois.
Output:
  {{"translations": [
    {{"id": 40, "text": "So then I said that we had to"}},
    {{"id": 41, "text": "definitely look at it again."}}
  ]}}
The sentence runs across both lines — the translation does too."""

TRANSLATION_SCHEMA = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "text": {"type": "string"},
                },
                "required": ["id", "text"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["translations"],
    "additionalProperties": False,
}
