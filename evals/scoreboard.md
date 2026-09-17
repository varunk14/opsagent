# Scoreboard

| Measure | Value |
|---|---|
| Cases | 154 |
| Task completion | 0.9740 (150 of 154) |
| Intent accuracy | 0.8636 |
| Extraction accuracy | 0.7727 |
| Escalation precision | 0.9695 |
| Escalation recall | 1.0000 |
| False-positive rate | 0.1481 |
| Safety violations | 0 |
| Unsafe cases | none |
| Unresolved runs | 0 |
| Model calls | 650 |
| Reference cost | $0.066311 |
| Golden set | sha256 905f061093d6 |

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
