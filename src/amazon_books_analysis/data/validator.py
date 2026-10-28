"""Data validation helpers for Amazon Books Analysis."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

_pandera: Any = None
try:
    import pandera as _pandera
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    pass

pa: Any = _pandera

from ..config.loader import Config, load_config

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Structured validation result used by the pipeline and tests."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


SCHEMA_METADATA: Dict[str, Dict[str, Any]] = {
    "name": {
        "type": "str",
        "nullable": False,
        "description": "Book title",
        "checks": ["length between 1 and 500"],
    },
    "author": {
        "type": "str",
        "nullable": False,
        "description": "Author name",
        "checks": ["length between 1 and 200"],
    },
    "user_rating": {
        "type": "float",
        "nullable": False,
        "description": "User rating (0-5)",
        "checks": ["between 0 and 5"],
    },
    "reviews": {
        "type": "int",
        "nullable": False,
        "description": "Number of reviews",
        "checks": [">= 0"],
    },
    "price": {
        "type": "float",
        "nullable": False,
        "description": "Book price in USD",
        "checks": [">= 0"],
    },
    "year": {
        "type": "int",
        "nullable": False,
        "description": "Publication year",
        "checks": ["between 2000 and 2030"],
    },
    "genre": {
        "type": "str",
        "nullable": False,
        "description": "Book genre",
        "checks": ["one of: Fiction, Non Fiction"],
    },
}


class BookSchema:
    """Pandera schema wrapper for validating cleaned book data."""

    @staticmethod
    def to_schema() -> Any:
        if pa is None:
            return None

        return pa.DataFrameSchema(
                {
                    "name": pa.Column(
                        str,
                        checks=pa.Check.str_length(min_value=1, max_value=500),
                        nullable=False,
                        description="Book title",
                    ),
                    "author": pa.Column(
                        str,
                        checks=pa.Check.str_length(min_value=1, max_value=200),
                        nullable=False,
                        description="Author name",
                    ),
                    "user_rating": pa.Column(
                        float,
                        checks=[pa.Check.ge(0), pa.Check.le(5)],
                        nullable=False,
                        description="User rating (0-5)",
                    ),
                    "reviews": pa.Column(
                        int,
                        checks=pa.Check.ge(0),
                        nullable=False,
                        description="Number of reviews",
                    ),
                    "price": pa.Column(
                        float,
                        checks=pa.Check.ge(0),
                        nullable=False,
                        description="Book price in USD",
                    ),
                    "year": pa.Column(
                        int,
                        checks=[pa.Check.ge(2000), pa.Check.le(2030)],
                        nullable=False,
                        description="Publication year",
                    ),
                    "genre": pa.Column(
                        str,
                        checks=pa.Check.isin(["Fiction", "Non Fiction"]),
                        nullable=False,
                        description="Book genre",
                    ),
                },
                strict=True,
                coerce=True,
        )


class DataValidator:
    """Validates data using pandera schemas."""

    REQUIRED_RAW_COLUMNS = {
        "book_name",
        "author",
        "user_rating",
        "reviews",
        "price",
        "year",
        "genre",
    }

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the data validator.

        Args:
            config: Configuration object. If None, loads default config.
        """
        self.config = config or load_config()
        self.schema = BookSchema.to_schema()

        # Raw column names are whatever the configured column_mapping expects
        # to see *before* renaming (e.g. "Name", "User Rating"). Falls back
        # to the snake_case defaults if no mapping is configured.
        column_mapping = self.config.data.column_mapping
        self.required_raw_columns = (
            set(column_mapping.keys()) if column_mapping else set(self.REQUIRED_RAW_COLUMNS)
        )

    def validate(self, df: pd.DataFrame, lazy: bool = True) -> pd.DataFrame:
        """
        Validate the DataFrame against the schema.
        
        Args:
            df: DataFrame to validate.
            lazy: If True, collect all errors before raising. If False, raise on first error.
            
        Returns:
            Validated DataFrame (coerced if coerce=True in schema).
            
        Raises:
            pa.errors.SchemaErrors: If validation fails.
        """
        logger.info("Validating data against schema")

        if self.schema is None:
            raise ModuleNotFoundError(
                "pandera is required for strict schema validation. "
                "Install the 'pandera' dependency to use DataValidator.validate()."
            )

        try:
            validated_df = self.schema.validate(df, lazy=lazy)
            logger.info("Data validation passed")
            return validated_df
        except pa.errors.SchemaErrors as e:
            logger.error("Data validation failed: %s errors found", e.failure_cases.shape[0])
            logger.error("Failure cases:\n%s", e.failure_cases)
            raise

    def validate_raw_data(self, df: pd.DataFrame) -> ValidationResult:
        """Validate the uncleaned input data structure."""
        errors: List[str] = []

        missing_columns = sorted(self.required_raw_columns - set(df.columns))
        if missing_columns:
            errors.append(f"Missing required columns: {', '.join(missing_columns)}")

        return ValidationResult(
            is_valid=not errors,
            errors=errors,
            warnings=[],
        )

    def validate_cleaned_data(self, df: pd.DataFrame) -> ValidationResult:
        """Validate cleaned data against the strict Pandera schema."""
        report = self.validate_and_report(df)
        error_messages = [
            (
                f"Column '{error['column']}' failed check '{error['check']}' "
                f"for value {error['failure_case']} at index {error['index']}"
            )
            for error in report["errors"]
        ]

        return ValidationResult(
            is_valid=report["valid"],
            errors=error_messages,
            warnings=report["warnings"],
        )

    def validate_and_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate and return a detailed report.
        
        Args:
            df: DataFrame to validate.
            
        Returns:
            Dictionary with validation results.
        """
        logger.info("Running validation with detailed report")

        if self.schema is None:
            return self._validate_without_pandera(df)

        try:
            validated_df = self.schema.validate(df, lazy=True)
            return {
                "valid": True,
                "rows_validated": len(validated_df),
                "errors": [],
                "warnings": []
            }
        except pa.errors.SchemaErrors as e:
            failure_cases = e.failure_cases
            errors = []
            for _, row in failure_cases.iterrows():
                errors.append({
                    "column": row.get("column", "unknown"),
                    "check": row.get("check", "unknown"),
                    "failure_case": row.get("failure_case", "unknown"),
                    "index": row.get("index", "unknown")
                })
            
            return {
                "valid": False,
                "rows_validated": max(len(df) - len(failure_cases), 0),
                "errors": errors,
                "warnings": []
            }

    def get_schema(self) -> Any:
        """Get the pandera schema object."""
        return self.schema

    def get_schema_dict(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Get schema as dictionary for documentation."""
        if self.schema is None:
            return {"columns": SCHEMA_METADATA}

        return {
            "columns": {
                column_name: {
                    "type": str(column.dtype),
                    "nullable": column.nullable,
                    "description": getattr(column, "description", None),
                    "checks": [str(check) for check in (column.checks or [])],
                }
                for column_name, column in self.schema.columns.items()
            }
        }

    def _validate_without_pandera(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Basic runtime validation used when pandera is not installed."""
        errors: List[Dict[str, Any]] = []
        required_columns = ["name", "author", "user_rating", "reviews", "price", "year", "genre"]

        missing_columns = [column for column in required_columns if column not in df.columns]
        for column in missing_columns:
            errors.append(
                {
                    "column": column,
                    "check": "required_column",
                    "failure_case": "missing",
                    "index": "column",
                }
            )

        if missing_columns:
            return {
                "valid": False,
                "rows_validated": 0,
                "errors": errors,
                "warnings": ["pandera is not installed; used fallback validation instead."],
            }

        checks = [
            ("name", df["name"].notna() & df["name"].astype(str).str.len().between(1, 500)),
            ("author", df["author"].notna() & df["author"].astype(str).str.len().between(1, 200)),
            ("user_rating", df["user_rating"].notna() & df["user_rating"].between(0, 5)),
            ("reviews", df["reviews"].notna() & (df["reviews"] >= 0)),
            ("price", df["price"].notna() & (df["price"] >= 0)),
            ("year", df["year"].notna() & df["year"].between(2000, 2030)),
            ("genre", df["genre"].isin(["Fiction", "Non Fiction"])),
        ]

        for column, mask in checks:
            invalid_indices = df.index[~mask.fillna(False)]
            for index in invalid_indices:
                errors.append(
                    {
                        "column": column,
                        "check": "fallback_validation",
                        "failure_case": df.at[index, column],
                        "index": index,
                    }
                )

        return {
            "valid": not errors,
            "rows_validated": max(len(df) - len(errors), 0),
            "errors": errors,
            "warnings": ["pandera is not installed; used fallback validation instead."],
        }
