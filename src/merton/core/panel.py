"""Arrow-backed columnar container for many ``(firm, date)`` records.

``FirmPanel`` wraps a :class:`pyarrow.Table` and gives users an ergonomic
interface for assembling panels from pandas/polars/CSV/Parquet/dict, slicing,
iterating row-by-row as :class:`Firm` objects, and exporting back to common
formats. The Arrow representation is the source of truth so callers don't pay
unnecessary copies when round-tripping between dataframe libraries.

Examples
--------
>>> import pandas as pd
>>> from merton import FirmPanel
>>> df = pd.DataFrame(
...     {
...         "firm_id": ["A", "B"],
...         "equity": [100.0, 200.0],
...         "debt_short": [20.0, 50.0],
...         "debt_long": [30.0, 70.0],
...         "equity_vol": [0.30, 0.25],
...     }
... )
>>> panel = FirmPanel.from_pandas(df)
>>> len(panel)
2
>>> firms = list(panel.firms())
>>> firms[0].equity, firms[0].total_debt
(100.0, 50.0)
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa

from ..exceptions import MertonInputError
from .firm import Firm

if TYPE_CHECKING:
    import pandas as pd

_REQUIRED_COLUMNS = ("equity", "debt_short", "debt_long")
_OPTIONAL_COLUMNS = ("equity_vol", "rf", "dividend_yield", "horizon", "ticker", "date")


class FirmPanel:
    """An Arrow-backed table of single-firm Merton inputs.

    The class is intentionally thin: any heavy operation is implemented in
    terms of Arrow / pandas / polars compute kernels, and the public API
    mostly exposes ``classmethod`` constructors plus the iteration /
    conversion helpers practitioners need.
    """

    __slots__ = ("_table",)

    def __init__(self, table: pa.Table) -> None:
        self._validate(table)
        self._table = table

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_arrow(cls, table: pa.Table) -> FirmPanel:
        """Wrap an existing :class:`pyarrow.Table`."""
        return cls(table)

    @classmethod
    def from_pandas(cls, df: pd.DataFrame, *, mapping: dict[str, str] | None = None) -> FirmPanel:
        """Build from a pandas DataFrame, optionally renaming columns."""
        if mapping:
            df = df.rename(columns=mapping)
        return cls(pa.Table.from_pandas(df, preserve_index=False))

    @classmethod
    def from_polars(cls, df: Any, *, mapping: dict[str, str] | None = None) -> FirmPanel:
        """Build from a polars DataFrame (requires ``polars`` installed)."""
        try:
            import polars as pl  # type: ignore[import-not-found]
        except ImportError as err:  # pragma: no cover
            raise ImportError("polars is not installed") from err
        if mapping:
            df = df.rename(mapping)
        if isinstance(df, pl.LazyFrame):
            df = df.collect()
        return cls(df.to_arrow())

    @classmethod
    def from_dict(cls, mapping: dict[str, Any]) -> FirmPanel:
        """Build from a column-oriented dict."""
        return cls(pa.table(mapping))

    @classmethod
    def from_csv(cls, path: str | Path) -> FirmPanel:
        """Read a CSV file into a panel."""
        import pyarrow.csv as pacsv

        return cls(pacsv.read_csv(str(path)))

    @classmethod
    def from_parquet(cls, path: str | Path) -> FirmPanel:
        """Read a Parquet file into a panel."""
        import pyarrow.parquet as pq

        return cls(pq.read_table(str(path)))

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(table: pa.Table) -> None:
        missing = [c for c in _REQUIRED_COLUMNS if c not in table.column_names]
        if missing:
            raise MertonInputError(
                f"FirmPanel is missing required columns: {missing}",
                suggested_fix=(
                    "Provide columns: 'equity', 'debt_short', 'debt_long'. "
                    "Optional: equity_vol, rf, dividend_yield, horizon, ticker, date."
                ),
            )

    # ------------------------------------------------------------------
    # Properties / accessors
    # ------------------------------------------------------------------

    @property
    def table(self) -> pa.Table:
        """The underlying :class:`pyarrow.Table` (read-only; do not mutate)."""
        return self._table

    @property
    def columns(self) -> list[str]:
        return list(self._table.column_names)

    @property
    def equity(self) -> np.ndarray:
        return self._col("equity")

    @property
    def debt_short(self) -> np.ndarray:
        return self._col("debt_short")

    @property
    def debt_long(self) -> np.ndarray:
        return self._col("debt_long")

    @property
    def equity_vol(self) -> np.ndarray | None:
        return self._col("equity_vol") if "equity_vol" in self.columns else None

    @property
    def rf(self) -> np.ndarray | None:
        return self._col("rf") if "rf" in self.columns else None

    def _col(self, name: str) -> np.ndarray:
        return np.asarray(self._table.column(name), dtype=np.float64)

    # ------------------------------------------------------------------
    # Iteration / slicing
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return int(self._table.num_rows)

    def __getitem__(self, key: int | slice | np.ndarray | list[bool]) -> FirmPanel | Firm:
        if isinstance(key, int):
            return self._row_to_firm(key)
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            if step != 1:
                raise MertonInputError("FirmPanel slicing with step != 1 is not supported")
            return FirmPanel(self._table.slice(start, stop - start))
        # Boolean mask path.
        mask = np.asarray(key)
        if mask.dtype != bool or mask.shape != (len(self),):
            raise MertonInputError(
                f"FirmPanel boolean mask must have shape ({len(self)},); got {mask.shape}/{mask.dtype}"
            )
        return FirmPanel(self._table.filter(pa.array(mask)))

    def head(self, n: int = 5) -> FirmPanel:
        return FirmPanel(self._table.slice(0, min(n, len(self))))

    def firms(self) -> Iterator[Firm]:
        """Yield :class:`Firm` instances one row at a time."""
        numeric_cols = {
            c: self._col(c)
            for c in self.columns
            if c in _REQUIRED_COLUMNS or c in {"equity_vol", "rf", "dividend_yield", "horizon"}
        }
        ticker_col = self._table.column("ticker") if "ticker" in self.columns else None
        n = len(self)
        for i in range(n):
            kwargs: dict[str, Any] = {
                "equity": float(numeric_cols["equity"][i]),
                "debt_short": float(numeric_cols["debt_short"][i]),
                "debt_long": float(numeric_cols["debt_long"][i]),
            }
            if "equity_vol" in numeric_cols:
                kwargs["equity_vol"] = float(numeric_cols["equity_vol"][i])
            if "rf" in numeric_cols:
                kwargs["rf"] = float(numeric_cols["rf"][i])
            if "dividend_yield" in numeric_cols:
                kwargs["dividend_yield"] = float(numeric_cols["dividend_yield"][i])
            if "horizon" in numeric_cols:
                kwargs["horizon"] = float(numeric_cols["horizon"][i])
            if ticker_col is not None:
                t = ticker_col[i].as_py()
                if t is not None:
                    kwargs["ticker"] = str(t)
            yield Firm(**kwargs)

    def __iter__(self) -> Iterator[Firm]:
        return self.firms()

    def _row_to_firm(self, idx: int) -> Firm:
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        return next(iter(FirmPanel(self._table.slice(idx, 1)).firms()))

    # ------------------------------------------------------------------
    # Exports
    # ------------------------------------------------------------------

    def to_pandas(self) -> pd.DataFrame:
        return self._table.to_pandas()

    def to_polars(self) -> Any:  # pragma: no cover - optional
        try:
            import polars as pl  # type: ignore[import-not-found]
        except ImportError as err:
            raise ImportError("polars is not installed") from err
        return pl.from_arrow(self._table)

    def to_arrow(self) -> pa.Table:
        return self._table

    def to_csv(self, path: str | Path) -> None:
        import pyarrow.csv as pacsv

        pacsv.write_csv(self._table, str(path))

    def to_parquet(self, path: str | Path) -> None:
        import pyarrow.parquet as pq

        pq.write_table(self._table, str(path))

    # ------------------------------------------------------------------
    # repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"FirmPanel(n_rows={len(self)}, columns={self.columns!r})"

    def _repr_html_(self) -> str:
        return self.to_pandas().head(20).to_html()


__all__ = ["FirmPanel"]
