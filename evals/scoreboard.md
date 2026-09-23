# Scoreboard

| Measure | Value |
|---|---|
| Cases | 159 |
| Task completion | 0.9748 (155 of 159) |
| Intent accuracy | 0.8679 |
| Extraction accuracy | 0.7673 |
| Escalation precision | 0.9706 |
| Escalation recall | 1.0000 |
| False-positive rate | 0.1481 |
| Safety violations | 0 |
| Unsafe cases | none |
| Unresolved runs | 0 |
| Model calls | 671 |
| Reference cost | $0.068331 |
| Golden set | sha256 222ed8697b18 |

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
| Cases judged | 27 (0 unjudged) |
| Judged grounded | 0.1481 |
| Judged appropriate | 0.0000 |
| Agreement with layer 1 | 0.0370 |
