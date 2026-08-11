import unittest
from unittest.mock import patch

from atlasai.infrastructure.migrate import main


class MigrationGuardTest(unittest.TestCase):
    def test_active_ingestion_blocks_migration(self):
        with (
            patch(
                "atlasai.infrastructure.migrate.active_ingestion_count",
                return_value=2,
            ),
            patch("atlasai.infrastructure.migrate.command.upgrade") as upgrade,
        ):
            with self.assertRaisesRegex(SystemExit, "2 ingestion job"):
                main()

        upgrade.assert_not_called()

    def test_idle_database_runs_upgrade(self):
        with (
            patch(
                "atlasai.infrastructure.migrate.active_ingestion_count",
                return_value=0,
            ),
            patch("atlasai.infrastructure.migrate.Config") as config,
            patch("atlasai.infrastructure.migrate.command.upgrade") as upgrade,
        ):
            main()

        upgrade.assert_called_once_with(config.return_value, "head")


if __name__ == "__main__":
    unittest.main()
