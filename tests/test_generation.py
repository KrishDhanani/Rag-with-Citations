import pytest

from src.generation.refusal import should_refuse


@pytest.mark.parametrize(
    ("context", "score", "expected"),
    [
        ("[1] Nicolas Flamel is a noted alchemist.", 0.997, False),
        ("[1] Harry said he was boiling.", 0.000017, True),
        ("   ", 0.997, True),
        ("[1] Retrieved without reranking.", None, True),
        ("[1] Evidence at the threshold.", 0.5, False),
        ("[1] Invalid score.", float("nan"), True),
    ],
)
def test_should_refuse(context, score, expected):
    assert should_refuse(context, score, threshold=0.5) is expected
