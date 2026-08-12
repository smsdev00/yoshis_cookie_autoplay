# Baseline and experiment protocol

## Stable references

- `reference/stage4-pre-docker` is the immutable known-good reference. It points
  to the pre-Docker autoplay validated reaching stage 4 with the host bsnes,
  `uinput`, visual cursor navigation, and strict post-move verification.
- `baseline/stable` is the best accepted implementation. Experiments start from
  this branch and merge back only after passing this protocol.
- Development happens in focused branches such as
  `experiment/cookie-variant-001` or `experiment/cursor-state-002`.

Git does not provide read-only local branches. The tag must never be moved or
recreated. On the remote, protect `baseline/stable` from direct pushes and force
pushes, require pull requests and required benchmark checks, and deny tag
updates/deletion for `reference/*`.

## Known-good environment

The reference is reproduced with:

- commit `62e58e0`;
- `/home/sms/Documents/bsnes-nightly/bsnes` and the adjacent `settings.bml`;
- host execution, not Docker;
- `PersistentUInputBackend`;
- screenshots in `/home/sms/Downloads`;
- the unmodified `kiosk --launch --yes-really-execute` command;
- no HUD, history database, online learning, fast navigation, or watchdog.

Changing any environmental item creates an experiment and must be evaluated as
such. A source checkout alone does not reproduce the reference environment.

## Microstate discovery loop

1. Create one experimental branch from `baseline/stable`.
2. Run the unchanged autoplay until its safety boundary stops it.
3. Preserve the framebuffer and context: previous board, cursor, proposed move,
   stage, timestamp, exception, and post-move frame when available.
4. Classify the failure as perception, execution, transition, or unknown.
5. Implement the smallest change that recognizes that single microstate.
6. Add the triggering capture as a permanent regression fixture.
7. Run the complete fixture suite. Previously known microstates must not regress.
8. Run one complete real game. Stop and reject the experiment on any new
   desynchronization.
9. Once locally sound, run the 20-game promotion benchmark.
10. Merge only if the promotion criteria pass. Otherwise preserve the report
    and reject or revise the experiment.

Unknown or low-confidence states are never guessed. The bot must stop before a
move, save evidence, and wait for a tested rule.

## Twenty-game promotion benchmark

Run two isolated groups under the same host, ROM, bsnes settings, timing, and
initial options:

- 20 games from `baseline/stable` (control);
- 20 games from the experimental commit (candidate).

Every game starts from a fresh ROM launch. Do not select Continue after Game
Over. Do not share mutable learning state between games or groups. Randomized
inputs, if introduced later, must use the same recorded seeds in both groups.

The benchmark recorder is passive: it may observe verified events but cannot
select moves, move the cursor, recover the game, or alter timing. Recorder
failure must not affect gameplay.

Record per game:

- highest stage reached;
- score when reliable, with its source identified;
- verified moves and moves per completed stage;
- cleared lines or cells when reliably observable;
- unknown microstates and detector failures;
- cursor-verification failures;
- post-move desynchronizations;
- termination reason and elapsed time.

Report at least mean, median, minimum, maximum, and success rate for reaching
stage 4 or higher. Keep raw per-game records with the summary.

## Promotion criteria

All mandatory safeguards must pass:

- zero new post-move desynchronizations;
- zero regressions in the complete microstate fixture corpus;
- no lower mean stage;
- no lower rate of reaching stage 4;
- no increase in technical failure rate.

The candidate must also show a clear improvement in at least one declared
primary target, such as mean/highest stage, reliable score, moves per stage, or
microstate coverage. Metrics and acceptance thresholds must be declared before
running the candidate benchmark.

A five-move smoke test, one successful game, or higher move count is not
evidence of improvement. If results are ambiguous, the candidate is not
promoted and the benchmark is repeated only after explaining why more evidence
is required.

## Promotion and rollback

Merge the exact benchmarked candidate commit into `baseline/stable`; do not add
unbenchmarked changes during promotion. Store the benchmark report in the
repository and record the candidate commit, environment fingerprint, outcome,
and reason for acceptance or rejection.

`reference/stage4-pre-docker` remains unchanged even after the baseline
improves. Rollback means returning `baseline/stable` to the last accepted commit
through a normal revert or replacement branch reviewed against the saved
reports; never rewrite or delete the reference tag.

## Experiment log

| Date | Candidate | Single change | Control result | Candidate result | Decision |
|---|---|---|---|---|---|
| 2026-08-12 | Initial baseline | Pre-Docker reference | Reached stage 4; stopped safely on unknown cookie after 23 verified moves | Same commit | Accepted |
