"""Reporting environment services."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from platform import python_version

from pandas import DataFrame

from app.domain.models import EnvironmentEntry


class EnvironmentReporter:
    """Build a typed table describing the execution environment."""

    @staticmethod
    def package_version(package_name: str) -> str:
        """Return an installed package version.

        :param package_name: Package name understood by importlib metadata.
        :type package_name: str
        :return: Installed version or ``unknown`` when unavailable.
        :rtype: str
        """
        try:
            installed_version: str = version(package_name)
            return installed_version
        except PackageNotFoundError:
            return "unknown"

    @classmethod
    def build_table(cls) -> DataFrame:
        """Build the environment table.

        :return: Environment table.
        :rtype: DataFrame
        """
        entries: list[EnvironmentEntry] = [
            EnvironmentEntry(component="Python", version=python_version()),
            EnvironmentEntry(
                component="birdnet",
                version=cls.package_version("birdnet"),
            ),
            EnvironmentEntry(
                component="onnxruntime",
                version=cls.package_version("onnxruntime"),
            ),
            EnvironmentEntry(
                component="numpy",
                version=cls.package_version("numpy"),
            ),
            EnvironmentEntry(
                component="pandas",
                version=cls.package_version("pandas"),
            ),
            EnvironmentEntry(
                component="pydantic",
                version=cls.package_version("pydantic"),
            ),
            EnvironmentEntry(
                component="scipy",
                version=cls.package_version("scipy"),
            ),
        ]
        records: list[dict[str, object]] = [
            entry.model_dump() for entry in entries
        ]
        table: DataFrame = DataFrame(records)
        return table
