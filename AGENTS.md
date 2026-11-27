# Project development

- Reproduce a reported failure and fix its root cause. Use reported prompts as regression cases, not as phrases to hard-code.
- Ground architectural changes in primary research or published engineering practice. Prefer the smallest design that handles the general behavior.
- Treat an example as evidence about the pipeline. Fix the shared context, tool, or evidence boundary that caused it before adding a new route.
- Check routing changes on the configured model across multiple turns, including topic switches, requests that need a tool, and ordinary conversation that should stay in chat.
