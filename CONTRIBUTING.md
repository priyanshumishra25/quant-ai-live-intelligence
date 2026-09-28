# Contributing

Keep changes reproducible, point-in-time aware, and testable. Model or feature changes should include a focused test and document when each input would have been available in real time. Tests must not require paid services or hidden network access.

Before opening a change:

```bash
make verify
```

If you alter the bundled deterministic fixture or lightweight artefact format, regenerate it with:

```bash
make bootstrap
make verify
```

Never commit provider secrets, proprietary raw datasets, real user data, or trained artefacts that cannot legally be redistributed.
