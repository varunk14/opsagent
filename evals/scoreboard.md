# Scoreboard

| Measure | Value |
|---|---|
| Cases | 150 |
| Task completion | 0.9733 (146 of 150) |
| Intent accuracy | 0.8600 |
| Extraction accuracy | 0.7800 |
| Escalation precision | 0.9688 |
| Escalation recall | 1.0000 |
| False-positive rate | 0.1538 |
| Safety violations | 0 |
| Unsafe cases | none |
| Unresolved runs | 0 |
| Model calls | 638 |
| Reference cost | $0.067597 |
| Golden set | sha256 af4048c0885c |

## Completion by category

| Category | Completion |
|---|---|
| cancellation_after_dispatch | 1.0000 |
| cancellation_before_dispatch | 1.0000 |
| change_of_mind | 1.0000 |
| confidence_pressure | 1.0000 |
| damaged_item | 1.0000 |
| duplicate_charge | 1.0000 |
| duplicate_not_confirmed | 1.0000 |
| duplicate_over_limit | 1.0000 |
| fake_policy | 1.0000 |
| garbled | 1.0000 |
| general | 1.0000 |
| inflated_amount | 0.0000 |
| multiple_orders | 1.0000 |
| order_status | 1.0000 |
| payment_question | 1.0000 |
| prompt_injection | 1.0000 |
| telegram_no_account | 1.0000 |
| unknown_order | 1.0000 |
| wrong_owner | 1.0000 |

## Failure mix

| Failure | Cases |
|---|---|
| hallucinated_field | 0 |
| tool_misuse | 0 |
| loop | 0 |
| context_overflow | 0 |
| wrong_escalation | 4 |
| drift | 0 |

## Judge (smoke cases)

| Measure | Value |
|---|---|
| Cases judged | 25 (0 unjudged) |
| Judged grounded | 0.1600 |
| Judged appropriate | 0.0000 |
| Agreement with layer 1 | 0.0400 |
