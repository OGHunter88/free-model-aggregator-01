# free-model-aggregator-01

Entry for the make.fun bounty **free-model-aggregator-01**: one server that serves the OpenAI
Chat Completions API and the Anthropic Messages API over ten simulated free-tier model providers,
routing each of 20 requests to the best free model that can serve it.

## How it works

- `main.ts` — the server (Node 22, standard library only). It reads the trial brief
  (`$MAKEFUN_PROMPT_FILE`), probes every provider's catalog route, registers its port and catalog
  with the env, serves the client's requests, and exits 0 when the env says the workload is done.
- `lib/providers.ts` — the brief's three documents (provider directory, quality table, workload
  manifest), the catalog probe in all three wires, and the catalog builder.
- `lib/wire.ts` — the client wires and the upstream wires: OpenAI ↔ Anthropic translation and the
  Gemini generateContent wire, in JSON and in server-sent events.

## Routing

- Candidates are ranked by the quality table's `rating_milli`; only free models that meet the
  request's requirements are ever picked.
- The first pick spreads inside a 15-milli band of the best rating by fewest calls so far, so no
  one provider's per-trial budget is spent on requests another provider answers nearly as well.
- On a 402, 429 or 5xx the request fails over to the next candidate by rating; a 402 marks that
  provider dead for the trial.
- JSON-mode requests never ride an anthropic wire, whose echo cannot carry JSON mode.
- Everything is deterministic: every choice comes from the brief, the requests and the providers'
  answers, so a replay leaves the same state export, byte for byte.

## Local grading

Practice trials, scored by the kit's own scorer:

```
node cli/makefun.mjs grade-local --kit . --bot <this repo> --driver process --instances 3
```

Mean over the 3 practice trials: **score 99** (quality 99, served 100, wire 100,
failover 100, catalog 100).

## License

MIT. See [LICENSE](LICENSE).
