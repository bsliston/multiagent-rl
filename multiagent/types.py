from collections import namedtuple

BatchMemories = namedtuple(
    "Batch",
    "state_t action_t reward_t state_tp1 action_tp1 done_t expected_return_t",
)
