"""Check how the graph uses the system clock and search evidence without a GPU."""

import json
import unittest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from assistant_agent.agent.graph_router import LangGraphAgent
from assistant_agent.agent.memory import ConversationMemory
from assistant_agent.llm.qwen import QwenAssistant
from assistant_agent.tools.internet_search_tool import SearchTool, SearchUnavailable


class SourceTests(unittest.TestCase):
    """Exercise the real graph with scripted model decisions and provider results."""

    def setUp(self):
        """Build an agent with isolated memory, inference, and search dependencies."""
        self.llm = Mock(spec=QwenAssistant)
        self.search = Mock(spec=SearchTool)
        self.memory = Mock(spec=ConversationMemory)
        self.memory.recent.return_value = []
        self.agent = LangGraphAgent(self.llm, self.search, self.memory)

    def test_chat_receives_trusted_runtime_context(self):
        """Give ordinary chat the machine clock alongside the conversation."""
        now = datetime(2026, 11, 27, 22, 0, tzinfo=timezone.utc)
        self.memory.recall.return_value = ([], [])
        self.llm.generate.side_effect = ['{"deliverable":"conversational_reply","query":""}', "Drafted note."]
        with patch("assistant_agent.agent.graph_router.datetime") as clock:
            clock.now.return_value = now
            self.agent.ask("Draft a note with the date and time I started.", "test")

        system = self.llm.generate.call_args.args[0]
        self.assertIn(f"Local datetime: {now.astimezone().isoformat()}", system)
        self.assertIn(f"UTC datetime: {now.isoformat()}", system)
        self.search.search.assert_not_called()

    def test_research_keeps_the_resolved_subject_during_retry(self):
        """Resolve a source-only follow-up and judge both lookups against the same question."""
        previous = [{"user": "How did Rosetta end its mission?", "reply": {"content": "It landed.", "sources": []}}]
        self.memory.recent.return_value = previous
        question = "How did ESA's Rosetta mission end?"
        retry_query = "ESA Rosetta controlled descent end of mission"
        self.llm.generate.side_effect = [
            '{"deliverable":"public_web_research","query":"online sources"}',
            json.dumps({"question": question}),
            json.dumps({"claims": [], "retry_query": retry_query}),
            json.dumps({"claims": [{
                "claim": "Rosetta made a controlled descent onto the comet.",
                "source_id": 2,
                "quote": "Rosetta made a controlled descent onto the comet",
            }], "retry_query": ""}),
        ]
        self.search.search.side_effect = [
            [{"href": "https://example.org/astronomy", "body": "Astronomy studies objects in space."}],
            [{"href": "https://example.org/rosetta", "body": "Rosetta made a controlled descent onto the comet."}],
        ]

        reply = self.agent.ask("Can you verify that online?", "test")

        self.assertEqual(self.llm.generate.call_args_list[1].kwargs["recent"], previous)
        self.assertEqual([call.args[0] for call in self.search.search.call_args_list], [question, retry_query])
        for call in self.llm.generate.call_args_list[2:]:
            self.assertEqual(json.loads(call.args[1])["question"], question)
        self.assertIn("controlled descent", reply["content"])
        self.assertEqual(reply["sources"][0]["url"], "https://example.org/rosetta")

    def test_search_without_a_subject_asks_for_one(self):
        """Avoid browsing when neither the request nor the chat supplies a topic."""
        self.llm.generate.side_effect = ['{"deliverable":"public_web_research","query":""}', '{"question":null}']

        reply = self.agent.ask("Use an online source.", "test")

        self.assertIn("What would you like me to look up", reply["content"])
        self.search.search.assert_not_called()

    def test_temporarily_unavailable_search_retries_the_resolved_question(self):
        """Keep the original subject through a transient provider failure."""
        question = "What is the current local date and time?"
        self.memory.recent.return_value = [
            {"user": question, "reply": {"content": "System clock reading.", "sources": []}},
        ]
        self.llm.generate.side_effect = [
            '{"deliverable":"public_web_research","query":""}', json.dumps({"question": question}),
        ]
        self.search.search.side_effect = SearchUnavailable("test timeout")

        reply = self.agent.ask("Check that on the web.", "test")

        self.assertEqual([call.args[0] for call in self.search.search.call_args_list], [question, question])
        self.assertIn("unavailable", reply["content"])

    def test_irrelevant_results_do_not_fall_back_to_the_follow_up_text(self):
        """Stop when evidence is irrelevant and the reviewer has no better query."""
        question = "What is the current local date and time?"
        self.llm.generate.side_effect = [
            '{"deliverable":"public_web_research","query":""}',
            json.dumps({"question": question}),
            '{"claims":[],"retry_query":""}',
        ]
        self.search.search.return_value = [
            {"href": "https://example.org/networks", "body": "The internet connects computers worldwide."},
        ]

        reply = self.agent.ask("Tell me based on internet information", "test")

        self.search.search.assert_called_once()
        self.assertEqual(reply["sources"], [])
        self.assertIn("enough reliable evidence", reply["content"])
