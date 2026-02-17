"""Tests for the VLM JSON parser.

Covers the three-tier parsing strategy:
1. Direct JSON parsing
2. Markdown code block extraction
3. First-{ to last-} substring extraction
Plus edge cases: garbage, empty, array, whitespace, nested JSON.
"""

from __future__ import annotations

import pytest

from oc_apprentice_worker.backends._json_parser import extract_json


class TestDirectJSON:
    def test_clean_json(self) -> None:
        result = extract_json('{"key": "value", "num": 42}')
        assert result == {"key": "value", "num": 42}

    def test_json_with_whitespace(self) -> None:
        result = extract_json('  \n  {"key": "value"}  \n  ')
        assert result == {"key": "value"}


class TestMarkdownCodeBlock:
    def test_json_code_block(self) -> None:
        text = 'Here is the result:\n```json\n{"target": "button"}\n```'
        result = extract_json(text)
        assert result == {"target": "button"}

    def test_no_language_code_block(self) -> None:
        text = 'Output:\n```\n{"target": "link"}\n```'
        result = extract_json(text)
        assert result == {"target": "link"}

    def test_code_block_with_extra_whitespace(self) -> None:
        text = '```json\n  {"a": 1}  \n```'
        result = extract_json(text)
        assert result == {"a": 1}


class TestSubstringExtraction:
    def test_preamble_text(self) -> None:
        text = 'Based on my analysis, the result is: {"target": "input", "confidence": 0.9}'
        result = extract_json(text)
        assert result["target"] == "input"
        assert result["confidence"] == 0.9

    def test_trailing_text(self) -> None:
        text = '{"target": "div"} I hope this helps!'
        result = extract_json(text)
        assert result == {"target": "div"}

    def test_preamble_and_trailing(self) -> None:
        text = 'Sure! {"key": "val"} Let me know if you need more.'
        result = extract_json(text)
        assert result == {"key": "val"}


class TestNestedJSON:
    def test_nested_objects(self) -> None:
        text = '{"outer": {"inner": "value"}, "list": [1, 2, 3]}'
        result = extract_json(text)
        assert result["outer"]["inner"] == "value"
        assert result["list"] == [1, 2, 3]


class TestMixedBraces:
    def test_unrelated_braces_before_json(self) -> None:
        text = 'The function f(x) = {x + 1} returns {"target": "button", "confidence": 0.8}'
        result = extract_json(text)
        assert result["target"] == "button"
        assert result["confidence"] == 0.8

    def test_multiple_unrelated_braces(self) -> None:
        text = 'if (x) { doA(); } else { doB(); } Result: {"key": "val"}'
        result = extract_json(text)
        assert result == {"key": "val"}


class TestErrorCases:
    def test_garbage_raises(self) -> None:
        with pytest.raises(ValueError, match="Could not extract"):
            extract_json("this is just random text with no json at all")

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="Empty input"):
            extract_json("")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValueError, match="Empty input"):
            extract_json("   \n\t  ")

    def test_array_raises(self) -> None:
        with pytest.raises(ValueError, match="Expected JSON object"):
            extract_json('[1, 2, 3]')

    def test_bare_string_raises(self) -> None:
        with pytest.raises(ValueError):
            extract_json('"just a string"')

    def test_incomplete_json_raises(self) -> None:
        with pytest.raises(ValueError, match="Could not extract"):
            extract_json('{"key": "value"')
