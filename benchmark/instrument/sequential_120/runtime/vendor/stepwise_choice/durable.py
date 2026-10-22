"""Retain exact-input journals without scoring backend failures as decisions."""
from stepwise_bench.__main__ import DurablePolicy
from stepwise_bench.policies import ResumeMismatchError

from .engine import MenuProtocolError


class DurableChoicePolicy(DurablePolicy):
    def next_action(self, public_view, history):
        try:
            return super().next_action(public_view, history)
        except ResumeMismatchError:
            raise
        except Exception as error:
            # With finite labels the model cannot generate a malformed command.
            # Missing menu options, context overflow and failed forward passes
            # are implementation/resource errors, including on cached replay.
            raise MenuProtocolError(f"Choice-interface execution failed: {error}") from error
