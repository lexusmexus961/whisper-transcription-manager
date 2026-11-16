"""The examples in core/diar_silence.py run — they name functions that exist.

Background: the speech_parts example once still called the function by an
old name — a doctest run failed with a NameError.
"""

import doctest

from whisper_tm.core import diar_silence


def test_the_module_examples_pass():
    result = doctest.testmod(diar_silence, verbose=False)
    assert result.attempted >= 3 and result.failed == 0
