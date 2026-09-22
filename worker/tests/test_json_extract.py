"""Tests for the VLM JSON parser.

Covers the three-tier parsing strategy:
1. Direct JSON parsing
2. Markdown code block extraction
3. First-{ to last-} substring extraction
Plus edge cases: garbage, empty, array, whitespace, nested JSON.
"""

from __future__ import annotations

import pytest

from agenthandover_worker.json_extract import clean_model_text, extract_json


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


class TestModelNoise:
    """Thinking blocks and chat-template end tokens (formerly stripped by
    five per-module parsers, now only here)."""

    def test_thinking_block_before_json(self) -> None:
        text = '<think>reasoning {"not": "this"}</think>\n{"key": "value"}'
        assert extract_json(text) == {"key": "value"}

    def test_thinking_block_and_fence(self) -> None:
        text = '<think>hmm</think>\n```json\n{"k": 1}\n```'
        assert extract_json(text) == {"k": 1}

    def test_several_thinking_blocks(self) -> None:
        text = '<think>a</think>{"k": 1}<think>b</think>'
        assert extract_json(text) == {"k": 1}

    def test_only_thinking_is_empty(self) -> None:
        with pytest.raises(ValueError, match="Empty input"):
            extract_json("<think>all reasoning, no answer</think>")

    @pytest.mark.parametrize("token", ["<|endoftext|>", "<|im_end|>", "<|end|>"])
    def test_end_tokens_cut_trailing_noise(self, token: str) -> None:
        text = '{"k": 1}' + token + 'Human: {"other": 2}'
        assert extract_json(text) == {"k": 1}

    def test_clean_model_text(self) -> None:
        assert clean_model_text("<think>x</think>  yes <|im_end|> junk") == "yes"


class TestTruncationRepair:
    """``repair_truncated=True``: output cut off at the token limit."""

    def test_off_by_default(self) -> None:
        # Without repair, a nested complete object is not returned in place
        # of the truncated whole.
        with pytest.raises(ValueError):
            extract_json('{"title": "T", "steps": [{"step": "a"}, {"st')

    def test_prose_braces_before_a_truncated_object(self) -> None:
        text = 'f(x) = {x + 1}. Answer: {"title": "T", "steps": ["a", "b'
        assert extract_json(text, repair_truncated=True) == {
            "title": "T", "steps": ["a"],
        }

    def test_drops_the_partial_value_and_closes_brackets(self) -> None:
        text = '{"title": "T", "steps": [{"step": "a"}, {"step": "b", "act'
        assert extract_json(text, repair_truncated=True) == {
            "title": "T", "steps": [{"step": "a"}, {"step": "b"}],
        }

    def test_unterminated_string_is_dropped(self) -> None:
        text = '{"a": 1, "b": "half a sent'
        assert extract_json(text, repair_truncated=True) == {"a": 1}

    def test_braces_inside_strings_do_not_count(self) -> None:
        text = '{"a": "x { [ y", "b": [1, 2'
        assert extract_json(text, repair_truncated=True) == {
            "a": "x { [ y", "b": [1],
        }

    def test_escaped_quotes(self) -> None:
        # The cut-off "b" object has no complete member, so "b" is dropped
        # rather than kept as an empty object.
        text = '{"a": "say \\"hi\\"", "b": {"c": tr'
        assert extract_json(text, repair_truncated=True) == {"a": 'say "hi"'}

    def test_inside_an_open_fence_after_thinking(self) -> None:
        text = '<think>plan</think>\n```json\n{"title": "T", "steps": ["a", "b'
        assert extract_json(text, repair_truncated=True) == {
            "title": "T", "steps": ["a"],
        }

    def test_complete_json_is_not_altered(self) -> None:
        assert extract_json('{"a": [1, 2]}', repair_truncated=True) == {"a": [1, 2]}

    def test_nothing_to_repair_still_raises(self) -> None:
        with pytest.raises(ValueError):
            extract_json("no json here", repair_truncated=True)
        with pytest.raises(ValueError):
            extract_json('{"', repair_truncated=True)
