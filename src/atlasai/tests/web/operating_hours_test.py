import unittest
from datetime import UTC, datetime

from atlasai.web.operating_hours import is_new_work_route, is_within_operating_hours


class OperatingHoursTest(unittest.TestCase):
    def test_accepts_work_inside_nairobi_window(self) -> None:
        self.assertTrue(
            is_within_operating_hours(datetime(2026, 8, 5, 8, 0, tzinfo=UTC))
        )

    def test_rejects_work_at_nairobi_close(self) -> None:
        self.assertFalse(
            is_within_operating_hours(datetime(2026, 8, 5, 19, 0, tzinfo=UTC))
        )

    def test_only_marks_compute_routes_as_new_work(self) -> None:
        self.assertTrue(is_new_work_route("POST", "/api/v1/documents"))
        self.assertTrue(
            is_new_work_route("POST", "/api/v1/threads/thread-id/messages")
        )
        self.assertFalse(is_new_work_route("GET", "/api/v1/session"))


if __name__ == "__main__":
    unittest.main()
