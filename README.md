# trufflepig

> Step-by-step RNA tumor analysis driven by [`pirlygenes`](https://github.com/pirl-unc/pirlygenes) gene sets.

## What this is

`trufflepig` subsumes the `analyze` command of `pirlygenes`, breaking it into many **composable sub-commands** with a **serializable record format** so each step's output can be streamed back to the user — suitable for building a website frontend that renders incremental results.

All backing data (gene sets, expression references, curated cancer-key-genes panels, surface proteins, therapy targets, etc.) continues to live in `pirlygenes` and is imported as a library. `trufflepig` is the orchestration, serialization, and CLI layer.

## Layout

```
trufflepig/
  cli.py            # sub-command entry points
  pipeline.py       # stage DAG (name -> upstream dependencies)
  workspace.py      # on-disk workspace format (records/ + figures/)
  stages/           # one module per stage; thin wrappers over pirlygenes
  version.py
```

## Stages

See `trufflepig list-stages` for the live DAG. Abbreviated:

```
load_expression    -> sample_context -> analyze -> decompose -> ranges
                                                        -> confidence
ranges, confidence -> render_targets, render_brief, render_provenance
```

Each stage reads the records it depends on from the workspace and writes
its own. A website backend can call any single stage and immediately
stream its output back to the frontend.

## Status

Scaffold only. Stage implementations are being migrated out of
`pirlygenes` in the issues tracked on this repo.

## License

Apache 2.0 — see [LICENSE](LICENSE).
