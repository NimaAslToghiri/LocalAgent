"""Run multi-turn regression checks against Signal's running local server."""

import json
import re
import unittest
from datetime import datetime, timezone

from websockets.sync.client import connect

from assistant_agent.config import PORT


class ConversationTests(unittest.TestCase):
    """Exercise real model decisions in a fresh conversation for each scenario."""

    def setUp(self):
        """Open a separate chat and register cleanup before sending any prompts."""
        host = f"127.0.0.1:{PORT}"
        self.ws = connect(f"ws://{host}/ws", origin=f"http://{host}", proxy=None)
        self.addCleanup(self.ws.close)
        self.ws.send(json.dumps({"type": "hello"}))
        ready = json.loads(self.ws.recv(timeout=60))
        self.assertEqual(ready["type"], "ready")
        self.assertEqual(ready["turns"], [])
        self.addCleanup(self.clear_chat)

    def clear_chat(self):
        """Delete only the new conversation issued to this test."""
        self.ws.send(json.dumps({"type": "clear"}))
        self.assertEqual(json.loads(self.ws.recv(timeout=60))["type"], "cleared")

    def ask(self, prompt):
        """Send one user turn and return the completed reply."""
        self.ws.send(json.dumps({"type": "message", "content": prompt}))
        reply = json.loads(self.ws.recv(timeout=60))
        if reply["type"] == "status":
            reply = json.loads(self.ws.recv(timeout=120))
        self.assertEqual(reply["type"], "response", reply)
        return reply

    def test_conversation_after_history(self):
        """Switch from transcript lookup to dialogue and then back to the first record."""
        first = "hey, can you tell me what was my previous message"
        self.assertIn("no earlier messages from you", self.ask(first)["content"])
        for prompt in ("perfect, ask my name", "ask my name"):
            reply = self.ask(prompt)
            self.assertFalse(reply.get("plain_text"))
            question = reply["content"].lower().replace("’", "'")
            self.assertRegex(question, r"(what[^?]*(your name|call you)|(?:tell|share|give)[^?]*your name)[^?]*\?")
        self.assertFalse(self.ask("My name is Mira.").get("plain_text"))
        answer = self.ask("What is my name?")
        self.assertFalse(answer.get("plain_text"))
        self.assertIn("mira", answer["content"].lower())
        transcript = self.ask("Quote my first message.")
        self.assertEqual(transcript["content"].split("You:\n")[-1], first)

    def test_topic_switch_and_return(self):
        """Check a different topic switch, a remembered preference, and an exact quote."""
        first = "Do our saved messages contain anything yet?"
        self.assertIn("no earlier", self.ask(first)["content"])
        question = self.ask("Ask which instrument I'd like to learn.")["content"].lower()
        self.assertIn("?", question)
        self.assertIn("instrument", question)
        self.ask("Violin.")
        reply = self.ask("Which instrument did I choose?")
        self.assertFalse(reply.get("plain_text"))
        self.assertIn("violin", reply["content"].lower())
        transcript = self.ask("Show me the very first thing I typed.")
        self.assertTrue(transcript.get("plain_text"))
        self.assertEqual(transcript["content"].split("You:\n")[-1], first)

    def test_current_clock_after_history(self):
        """Get the current UTC time after a transcript request in a fresh chat."""
        self.assertIn("no earlier", self.ask("What is my previous message?")["content"])
        before = datetime.now(timezone.utc).replace(microsecond=0)
        reply = self.ask("Tell me today's date and time in UTC.")
        reading = re.search(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2})", reply["content"])
        self.assertIsNotNone(reading, reply["content"])
        timestamp = datetime.fromisoformat(reading.group().replace("Z", "+00:00"))
        self.assertEqual(timestamp.utcoffset(), timezone.utc.utcoffset(None))
        self.assertLessEqual(before, timestamp)
        self.assertLessEqual(timestamp, datetime.now(timezone.utc))

    def test_transform_and_quote(self):
        """Transform prior content, then quote an assistant reply and count user turns."""
        self.ask("The English word for today is lighthouse.")
        translation = self.ask("Translate that word into Spanish.")["content"]
        self.assertIn("faro", translation.lower())
        transcript = self.ask("Repeat your last reply verbatim.")
        self.assertTrue(transcript.get("plain_text"))
        self.assertEqual(transcript["content"].split("Signal:\n")[-1], translation)
        self.assertIn("3 earlier messages from you", self.ask("How many messages have I sent?")["content"])


if __name__ == "__main__":
    unittest.main()
