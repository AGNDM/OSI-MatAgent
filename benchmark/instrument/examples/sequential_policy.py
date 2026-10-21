"""Minimal public-menu plugin: always choose the first offered option.

This demonstrates the interface, not a useful laboratory policy. The evaluator
scores one action after any command/argument selections, then calls choose
again with the same goal and updated native feedback in the public history.

Compute-job payload (inside Slurm on SCITAS)::

    python eval/sequential.py run --policy examples.sequential_policy:make_policy \
        --mode teacher_forced --limit 1 --output output/first-choice-demo
"""


class FirstChoicePolicy:
    privileged = False
    identity = {"method": "first-public-menu-index", "diagnostic": True}

    def choose(self, view, history, options):
        # A model backend would show view/history/options and map its selected
        # displayed label to this zero-based index. Private case specs and
        # reference actions never enter this callback.
        return 0


def make_policy(context):
    """The context contains protocol, mode and the fixed public parameter bank."""
    return FirstChoicePolicy()


factory = make_policy
