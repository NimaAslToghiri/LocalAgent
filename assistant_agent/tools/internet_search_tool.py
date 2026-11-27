from ddgs import DDGS

class SearchTool:

    @property
    def name(self):
        return "search"

    def search(self, query: str, max_results: int = 5):
        
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(
                query,
                max_results=max_results
            ):
                results.append(
                    {
                        "title": r.get("title", ""),
                        "body": r.get("body", ""),
                        "href": r.get("href", "")
                    }
                )

        return results