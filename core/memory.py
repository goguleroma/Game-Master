from typing import Any, Dict, List


class MemoryStore:
    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []

    def record(self, entry: Dict[str, Any]) -> None:
        self.records.append(entry)

    def last(self, count: int = 10) -> List[Dict[str, Any]]:
        return self.records[-count:]
