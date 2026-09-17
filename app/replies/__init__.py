"""
Replies: the word the agent owes a customer once a run has rested.

Nothing here talks to a model. A run's outcome chooses one of a fixed set of
messages, that message is rendered and frozen into an outbox row in the same
transaction as the outcome (app/replies/outbox.py), and a drain sends it after
the commit and marks it sent (app/replies/send.py). At-least-once, the same
ordering the poll channels keep.
"""
