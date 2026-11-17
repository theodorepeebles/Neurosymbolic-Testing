"""The 10 test puzzles shown by the replay demo, in display order (8 the pipeline got right,
then 2 it got wrong). Ids are run_id in data/sft_test.jsonl and problem_id in the source run."""

DEMO_PROBLEM_IDS = [
    "9b8a0c74bfb545078e9eda4b3d4ae946",  # 1  ordering
    "4aa1069f1c2242fa98c5cd0431404361",  # 2  grouping
    "377f775caa6845a393456005d98bbfc2",  # 3  knights_and_knaves
    "98112969763f44beaad54b69634965c1",  # 4  grouping + ordering
    "9c3be010ea734d5499d09ac3ae2ba8a7",  # 5  grouping + ordering, 4 choices
    "762c10e188b44379b0d358910ed1eb60",  # 6  knights_and_knaves + ordering
    "7ea3b58227244f7ca2fdec9c68415fd4",  # 7  grouping + knights_and_knaves
    "51a6632a11874c14aa82061e41f867e9",  # 8  all three, 13 rules
    "3470c07b758b4cbbb9f77b25c6a6ff3a",  # 9  wrong: no choice proven
    "d969d4b1162343f4882c990a12fd2c40",  # 10 wrong: two choices both pass
]
