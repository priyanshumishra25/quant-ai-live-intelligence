# v3.0.1 Free-tier patch

This patch hardens the live stock-intelligence workflow for strict Alpha Vantage free-tier quotas.

## Provider-call changes

- No live stock analysis runs during dashboard boot or refresh.
- Live providers are contacted only after an explicit **Analyze stock** action.
- Alpha Vantage daily history defaults to `outputsize=compact`; `full` is opt-in.
- Alpha Vantage requests are serialized and paced by `1.10s` by default.
- Explicit uppercase tickers (for example `AAPL`) skip `SYMBOL_SEARCH`.

## Layered caches

| Data | Default TTL |
| --- | ---: |
| Symbol/company resolution | 7 days |
| Daily market history | 6 hours |
| News | 30 minutes |
| Reddit retrieval | 10 minutes |
| Final live-intelligence result | 30 minutes |

The provider caches sit below the forecast cache. Changing from a 5-day to a 20-day forecast therefore reuses the same market/news evidence instead of spending additional upstream calls.

## Error handling

Alpha Vantage quota/rate-limit/premium responses are translated into concise provider-limit errors and returned by the API as HTTP `429` rather than a generic `502`.

## Correctness fixes

- Historical news outside the available compact price-history window is ignored rather than being collapsed onto the first market date.
- Tests explicitly neutralize local `.env` provider credentials so `make verify` never consumes live quota.
- Local `.env` files are allowed during verification as long as Git does not track them.

## Verification

The patch includes regression coverage for:

- provider-level caching across forecast horizons;
- explicit-ticker symbol-search skipping;
- free-tier `compact` daily-history requests;
- provider quota-message translation.
