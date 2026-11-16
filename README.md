# free-model-aggregator-01

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/badhanali30-bit/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/badhanali30-bit/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One OpenAI- and Anthropic-compatible endpoint that routes every request to the
best free model — no model name needed, no config, no paid tier.

The server reads the trial brief (ten providers, the sealed quality table, the
workload manifest), probes every provider's catalog route in the provider's own
wire, and registers the one catalog it found. Each of the 20 requests is then
routed to the best free model that meets its requirements, ranked by the
quality table's rating; a provider that answers 402 has spent its free
allowance and is never tried again, while a 429 or 5xx fails over to the next
best candidate, and calls spread over providers among equal ratings. Both
wires are served in JSON and in server-sent events (`POST /v1/chat/completions`
and `POST /v1/messages`, plus `GET /v1/models` in the shape of the auth header
it got), and the server exits 0 the moment the env says the workload is done.
`node main.ts --list` prints the same catalog document it registers, byte for
byte. Tested end to end against a mock trial: register, quality routing,
429 failover, Anthropic translation, SSE streaming and clean exit 0.

## Run it

```sh
node main.ts
```

```sh
node main.ts --list
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=06dntsta&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
