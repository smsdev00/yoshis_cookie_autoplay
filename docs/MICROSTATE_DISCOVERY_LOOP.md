# Microstate discovery loop

This guide defines the day-to-day workflow for extending autoplay perception
without changing behavior that already works. Read it together with
`docs/BASELINE_AND_EXPERIMENTS.md`.

## Goal

Let the known-good bot play until it reaches the next visual state it cannot
classify safely. Preserve that state, teach the detector only that case, add a
permanent regression test, and run again. Over time the fixture corpus becomes
a catalog of every relevant visual microstate.

The current solver is not the target of this loop. A perception failure is not
evidence that move selection needs changing.

## Invariants

- Start every experiment from `baseline/stable`.
- Use the host pre-Docker environment and
  `/home/sms/Documents/bsnes-nightly/bsnes`.
- Use `PersistentUInputBackend`, visual cursor navigation, and strict post-move
  verification.
- Never guess an unknown state or lower confidence merely to keep playing.
- Never move after an unknown cookie, ambiguous cursor, unknown screen, or
  failed post-move verification.
- Change one microstate rule per experimental branch.
- Every learned microstate requires a fixture and regression test.
- Do not introduce Docker, X11 input, HUD, persistence, online learning,
  watchdogs, or recovery changes while discovering perception microstates.

## Branch workflow

Create a focused branch from the accepted baseline:

```bash
git switch baseline/stable
git switch -c experiment/<microstate-name>
```

Use names that describe observed evidence, not a speculative fix, for example:

```text
experiment/occluded-heart-bottom-left
experiment/cursor-double-detection
experiment/stage4-transition-frame
```

Do not develop on `reference/stage4-pre-docker` or move that tag.

## Run the known-good bot

Run the original command without experimental flags:

```bash
venv/bin/python -m autoplay.kiosk kiosk \
  --launch \
  --yes-really-execute \
  --bsnes /home/sms/Documents/bsnes-nightly/bsnes \
  --rom "/home/sms/Downloads/Yoshi's Cookie (USA).zip"
```

Do not add move limits, alternate timings, alternate input backends, databases,
or recovery flags. A changed command is a different experiment.

Let the process stop at its existing safety boundary. Do not manually operate
the game during a run.

## Capture the failure packet

For every new stop, preserve a packet containing:

- the original framebuffer without resizing or recompression;
- the exception text and category;
- timestamp and stage;
- last verified board and cursor;
- proposed move, if one existed;
- frame immediately before input, if input was sent;
- expected and observed boards for post-move failures;
- exact commit and command used.

Use a stable case identifier such as:

```text
ms-0001-unknown-cookie-stage4
```

Store immutable image fixtures under a case-oriented path such as:

```text
tests/fixtures/microstates/ms-0001-unknown-cookie-stage4/frame.png
tests/fixtures/microstates/ms-0001-unknown-cookie-stage4/context.json
```

Never overwrite an older fixture with a newer capture.

## Classify before editing

Choose exactly one category:

- `perception-cookie`: cookie sprite is unknown or misclassified;
- `perception-cursor`: cursor is absent, duplicated, or occludes a cookie;
- `screen-transition`: menu, stage transition, Game Over, or animation state;
- `board-geometry`: extent, falling row, sparse component, or unexpected hole;
- `execution`: input was sent but the verified result is wrong;
- `unknown`: evidence is insufficient; collect more and do not patch yet.

Only the first four normally justify detector changes. Execution failures must
be investigated in input, cursor tracking, bindings, and timing without changing
the solver or weakening verification.

## Implement the smallest rule

A valid patch should:

- recognize the captured case;
- preserve all existing classifications;
- retain or improve confidence honestly;
- remain narrow enough to explain from pixel evidence;
- fail closed outside its proven range.

Avoid broad tolerance increases, fallback guesses, retry loops, or treating an
animation frame as a playable board. If multiple unrelated rules are needed,
split them into separate experiments.

## Add the regression fixture

The new test must load the preserved framebuffer and assert the full expected
outcome, for example:

- exact board array and cursor;
- explicit screen-state exception;
- explicit rejection of a non-playable animation;
- no input on an unknown or ambiguous state.

Also add a negative or neighboring case when the new rule could overlap an
existing sprite class. Run the complete suite, not only the new test:

```bash
venv/bin/python -m unittest discover -s tests -v
git diff --check
```

Do not continue to a real run while any prior fixture fails.

## Validate progression

After tests pass:

1. Start a fresh ROM using the exact known-good command.
2. Confirm the bot passes the newly learned microstate.
3. Let it continue until the next safety stop.
4. Confirm there was no desynchronization and no manual intervention.
5. Preserve the next failure packet before making another change.

One branch may contain one microstate fix and its tests. If the next stop is a
different microstate, finish or reject the current experiment before creating a
new branch.

## Promotion boundary

Microstate discovery establishes correctness and coverage; it does not by
itself prove overall performance. Before merging into `baseline/stable`, follow
the promotion requirements in `docs/BASELINE_AND_EXPERIMENTS.md`.

At minimum, reject immediately on:

- any new post-move desynchronization;
- any regression in the fixture corpus;
- lower confidence hidden by a guessed classification;
- failure to pass the triggering microstate in a fresh run.

For a candidate that changes runtime behavior or is intended for baseline
promotion, run the documented 20-game control/candidate benchmark.

## Required experiment record

Each experiment should leave a concise report containing:

```text
Case ID:
Branch and commit:
Baseline commit:
Exact command:
Failure category:
Observed evidence:
Minimal rule added:
Fixtures added:
Test result:
Fresh-run result:
Promotion decision:
```

The next developer should be able to reproduce the discovery and understand
why the rule exists without reading chat history.
