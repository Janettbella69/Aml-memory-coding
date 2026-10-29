# Script-style proxy (our own)

What ScriptMem asks of a memory system, without ScriptMem's data: its source scripts are not published and its public
questions must not be used for fitting (competition FAQ Q9). This folder holds a stand-in we built ourselves.

- **Corpus:** Bayard Veiller, *The Thirteenth Chair* (1916), Project Gutenberg eBook #14095, public domain in the USA.
  `parse_play.py` turns the Gutenberg text into ScriptMem's conversation schema (one session per act; `dialogue` lines
  with a speaker, `narration` lines with a null speaker; no timestamps): 1,422 lines, 121k characters, median dialogue
  line 47 characters. The text is downloaded at run time and is not stored in this repository.
- **Questions:** `questions.json`, 31 questions written by us in ScriptMem's public format — 19 single-choice (six options,
  the last one "Cannot infer …", gold in two), 8 multi-select (exact set), 4 ordering (exact sequence); the option list
  and the answer-format sentence sit inside the question text, as in ScriptMem. Each carries the `dia_id`s of the lines
  that settle it. `make_questions.py` is the source they are generated from.
- **Runner:** `scripts/eval_script_proxy.py` — Add in chunks of 20 messages, Search with `query` + `options`, the
  organisers' ScriptMem answer prompt and exact-letter scorer (imported from a clone of their repository).

Why it exists: with one-line turns, hit ± 2 blocks return about a third of a play that would fit into the answer window
whole. See `docs/METHOD_CARD.md` (0.8) for the numbers.
