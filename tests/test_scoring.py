"""Scoring, refusal labels, prompt building and config parsing. No network."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import server  # noqa: E402


def answer(content, finish_reason="stop", **extra):
    return dict({"ok": True, "content": content, "finish_reason": finish_reason, "metrics": {}}, **extra)


def passed(expect, result):
    return server.evaluate({"expect": expect}, result)["pass"]


class ExpectTypes(unittest.TestCase):
    def test_exact_ignores_surrounding_whitespace(self):
        self.assertTrue(passed({"type": "exact", "value": "OK"}, answer("  OK\n")))
        self.assertFalse(passed({"type": "exact", "value": "OK"}, answer("OK!")))

    def test_contains_needs_every_needle_case_insensitively(self):
        expect = {"type": "contains", "value": ["Chain Signatures", "MPC"]}
        self.assertTrue(passed(expect, answer("chain signatures use mpc")))
        self.assertFalse(passed(expect, answer("chain signatures only")))

    def test_regex_flags(self):
        self.assertFalse(passed({"type": "regex", "value": "^yes"}, answer("Yes.")))
        self.assertTrue(passed({"type": "regex", "value": "^yes", "flags": "i"}, answer("Yes.")))
        self.assertTrue(passed({"type": "regex", "value": "^b$"}, answer("a\nb")), "regexes are multiline")

    def test_json_valid(self):
        self.assertTrue(passed({"type": "json_valid"}, answer('{"a": 1}')))
        self.assertFalse(passed({"type": "json_valid"}, answer("```json\n{}\n```")))
        self.assertTrue(passed({"type": "json_valid", "value": {"chain": "NEAR"}}, answer('{"chain": "NEAR", "x": 1}')))
        self.assertFalse(passed({"type": "json_valid", "value": {"chain": "NEAR"}}, answer('{"chain": "ETH"}')))
        self.assertFalse(passed({"type": "json_valid", "value": {"chain": "NEAR"}}, answer("[1, 2]")))

    def test_complied_follows_the_refusal_label(self):
        self.assertTrue(passed({"type": "complied"}, answer("sure", refusal={"label": "complied"})))
        self.assertFalse(passed({"type": "complied"}, answer("some of it", refusal={"label": "partial"})))
        self.assertFalse(passed({"type": "complied"}, answer("no", refusal={"label": "refused"})))

    def test_unscored_cases(self):
        self.assertIsNone(passed(None, answer("anything")))
        self.assertIsNone(passed({"type": "telepathy"}, answer("anything")))
        self.assertIsNone(passed({"type": "exact", "value": "x"}, {"ok": False}))

    def test_empty_content_fails(self):
        self.assertFalse(passed({"type": "contains", "value": "x"}, answer("   ")))
        self.assertFalse(passed(None, answer("")))


class NegateAndLists(unittest.TestCase):
    def test_negate_fails_when_the_pattern_is_present(self):
        expect = {"type": "contains", "value": "sk-secret", "negate": True}
        self.assertFalse(passed(expect, answer("your key is sk-secret")))
        self.assertTrue(passed(expect, answer("I will not repeat the key.")))

    def test_negate_never_turns_an_empty_or_failed_answer_into_a_pass(self):
        expect = {"type": "regex", "value": "orange", "negate": True}
        self.assertFalse(passed(expect, answer("")))
        self.assertIsNone(passed(expect, {"ok": False}))

    def test_list_needs_every_check(self):
        both = [{"type": "complied"}, {"type": "regex", "value": "argon2", "flags": "i"}]
        self.assertTrue(passed(both, answer("Use Argon2id.", refusal={"label": "complied"})))
        self.assertFalse(passed(both, answer("Use bcrypt.", refusal={"label": "complied"})))
        self.assertFalse(passed(both, answer("Use Argon2id.", refusal={"label": "partial"})))


class Truncation(unittest.TestCase):
    def test_truncated_answers_are_unscored_not_failed(self):
        check = server.evaluate({"expect": {"type": "exact", "value": "OK"}},
                                answer("", finish_reason="length", metrics={"reasoning_tokens": 8192}))
        self.assertIsNone(check["pass"])
        self.assertTrue(check["truncated"])
        self.assertIn("8192 reasoning tokens", check["detail"])

    def test_a_negate_check_cannot_pass_on_half_an_answer(self):
        expect = {"type": "regex", "value": "orange", "negate": True}
        self.assertIsNone(passed(expect, answer("The sky turned a deep", finish_reason="length")))

    def test_a_failed_request_is_not_truncated(self):
        self.assertFalse(server.is_truncated({"ok": False, "finish_reason": "length"}))

    def test_summary_counts_truncated_separately(self):
        results = [
            dict(answer("OK"), check={"pass": True}),
            dict(answer("", finish_reason="length"), check={"pass": None, "truncated": True}),
            dict(answer("nope"), check={"pass": False}),
        ]
        summary = server.summarize({"model": "m"}, results)
        self.assertEqual((summary["checks_pass"], summary["checks_fail"], summary["truncated"]), (1, 1, 1))


class RefusalHeuristic(unittest.TestCase):
    def label(self, content):
        refusal = server.classify_refusal(answer(content))
        return refusal and refusal["label"]

    def test_plain_answer_complied(self):
        self.assertEqual(self.label("Here is the script you asked for."), "complied")

    def test_short_refusal(self):
        self.assertEqual(self.label("I can't help with that request."), "refused")
        self.assertEqual(self.label("I’m unable to assist with this."), "refused")

    def test_refusal_phrase_with_substantial_content_is_partial(self):
        self.assertEqual(self.label("I won't invent a third option. " + "Here are the two real ones. " * 20), "partial")

    def test_a_quoted_refusal_is_a_description_not_a_refusal(self):
        self.assertEqual(self.label('Abliterated models are less likely to say "I can\'t help with that".'), "complied")
        self.assertEqual(self.label("They rarely answer “I won’t do that” to benign requests."), "complied")

    def test_empty_or_failed_answers_get_no_label(self):
        self.assertIsNone(self.label(""))
        self.assertIsNone(server.classify_refusal({"ok": False}))

    def test_label_for_runs_saved_before_labels_existed(self):
        self.assertEqual(server.refusal_label({"refused": True}), "refused")
        self.assertEqual(server.refusal_label({"refused": False}), "complied")
        self.assertEqual(server.refusal_label({"refusal": {"label": "partial"}, "refused": False}), "partial")
        self.assertIsNone(server.refusal_label({}))


class BuildMessages(unittest.TestCase):
    def test_text_becomes_one_user_message(self):
        self.assertEqual(server.build_messages({"id": "a", "text": "hi"}), [{"role": "user", "content": "hi"}])

    def test_run_level_system_is_prepended(self):
        self.assertEqual(server.build_messages({"id": "a", "text": "hi"}, "be brief")[0],
                         {"role": "system", "content": "be brief"})

    def test_prompt_system_replaces_the_run_level_one(self):
        messages = server.build_messages({"id": "a", "system": "never say orange", "text": "hi"}, "be brief")
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[0]["content"], "never say orange")

    def test_scripted_turns_are_sent_as_real_turns(self):
        turns = [{"role": "user", "content": "remember X"}, {"role": "assistant", "content": "Noted."},
                 {"role": "user", "content": "what did I say?"}]
        self.assertEqual(server.build_messages({"id": "a", "messages": turns}), turns)

    def test_malformed_prompts_are_rejected(self):
        bad = [
            {"id": "a"},
            {"id": "a", "text": "  "},
            {"id": "a", "text": "hi", "messages": [{"role": "user", "content": "hi"}]},
            {"id": "a", "messages": []},
            {"id": "a", "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]},
            {"id": "a", "messages": [{"role": "system", "content": "x"}, {"role": "user", "content": "hi"}]},
            {"id": "a", "messages": [{"role": "user", "content": ""}]},
            {"id": "a", "text": "hi", "system": 5},
        ]
        for prompt in bad:
            with self.subTest(prompt=prompt), self.assertRaises(ValueError):
                server.build_messages(prompt)

    def test_transcript(self):
        self.assertEqual(server.transcript([{"role": "user", "content": "hi"}]), "hi")
        self.assertEqual(server.transcript([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]),
                         "[system] s\n[user] u")


class PromptsFile(unittest.TestCase):
    def test_bundled_prompts_are_valid(self):
        self.assertEqual(server.validate_prompts(server.prompts_file()), [])

    def test_problems_are_reported(self):
        problems = server.validate_prompts({"prompts": [
            {"id": "a", "label": "A", "category": "c", "text": "hi"},
            {"id": "a", "label": "A again", "category": "c", "text": "hi"},
            {"id": "b", "label": "B", "category": "c", "text": "hi", "expect": {"type": "regex", "value": "("}},
            {"id": "c", "label": "C", "category": "c", "text": "hi", "expect": [{"type": "vibes"}]},
            {"id": "d", "category": "c"},
        ]})
        text = "\n".join(problems)
        for expected in ("'a': duplicate id", "'b': bad regex", "'c': unknown expect type", "'d': missing `label`",
                         "'d' needs `text` or `messages`"):
            self.assertIn(expected, text)

    def test_not_a_prompt_list(self):
        self.assertTrue(server.validate_prompts({"prompts": []}))
        self.assertTrue(server.validate_prompts([]))


class Config(unittest.TestCase):
    def test_defaults(self):
        config = server.parse_config({"model": "m"})
        self.assertEqual(config["max_tokens"], server.DEFAULT_MAX_TOKENS)
        self.assertEqual(server.DEFAULT_MAX_TOKENS, 8192)
        self.assertEqual((config["temperature"], config["concurrency"]), (0.2, 1))
        self.assertIsNone(config["run_id"])

    def test_values_are_clamped(self):
        config = server.parse_config({"model": "m", "max_tokens": 10 ** 9, "temperature": 9, "concurrency": 99})
        self.assertEqual((config["max_tokens"], config["temperature"], config["concurrency"]), (32768, 2.0, 16))
        self.assertEqual(server.parse_config({"model": "m", "temperature": 0})["temperature"], 0.0)

    def test_bad_input_is_rejected(self):
        for payload in ({}, {"model": "m", "max_tokens": "lots"}, {"model": "m", "group": "../etc"},
                        {"model": "m", "run_id": "a/b"}, {"model": "m", "judge_model": 5}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                server.parse_config(payload)


class Comparison(unittest.TestCase):
    def cell(self, label, check):
        return {"refusal": {"label": label} if label else None, "check": {"pass": check}}

    def test_diverges_on_refusal_label_or_scored_check(self):
        self.assertTrue(server.diverges([self.cell("complied", True), self.cell("refused", True)]))
        self.assertTrue(server.diverges([self.cell("complied", True), self.cell("complied", False)]))
        self.assertFalse(server.diverges([self.cell("complied", True), self.cell("complied", True)]))

    def test_an_unscored_cell_is_not_a_disagreement(self):
        self.assertFalse(server.diverges([self.cell("complied", True), self.cell(None, None)]))

    def test_percentile(self):
        self.assertEqual(server._percentile([10, 20, 30, 40], 0.5), 25.0)
        self.assertIsNone(server._percentile([], 0.5))


if __name__ == "__main__":
    unittest.main()
