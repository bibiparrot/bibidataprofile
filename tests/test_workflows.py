import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import pytest

from bibidataprofile import workflows as wf


@pytest.fixture
def data():
    rng = np.random.default_rng(7)
    x = rng.normal(size=160)
    return pd.DataFrame({"x": x, "segment": rng.choice(["A", "B"], 160),
                         "regression": x * 3 + rng.normal(size=160),
                         "event": np.where(x > 0, "yes", "no"),
                         "date": pd.date_range("2025-01-01", periods=160)})


@pytest.mark.parametrize("suffix,separator,encoding", [("csv", ",", "utf-8-sig"), ("tsv", "\t", "gb18030")])
def test_text_import_preserves_columns_and_chinese(tmp_path, suffix, separator, encoding):
    source = pd.DataFrame({"姓名": ["王一", "张二"], "金额": [5, 8]})
    path = tmp_path / f"data.{suffix}"
    source.to_csv(path, sep=separator, encoding=encoding, index=False)
    pd.testing.assert_frame_equal(wf.load_data(path), source)
    pd.testing.assert_frame_equal(wf.load_upload(path.name, path.read_bytes()), source)


def test_database_query_and_empty_result(tmp_path):
    path = tmp_path / "data.db"
    with sqlite3.connect(path) as connection:
        pd.DataFrame({"value": [1, 2]}).to_sql("sample", connection, index=False)
    assert wf.query_database(f"sqlite:///{path.as_posix()}", "select value from sample")["value"].tolist() == [1, 2]
    with pytest.raises(ValueError, match="数据为空"):
        wf.query_database(f"sqlite:///{path.as_posix()}", "select * from sample where 1=0")


def test_types_are_partition_and_edits_are_validated(data):
    types = wf.infer_types(data)
    assert wf.parse_types(json.dumps(types), data.columns) == types
    types["Numeric"].append("event")
    with pytest.raises(ValueError, match="恰好"):
        wf.parse_types(json.dumps(types), data.columns)


def test_date_split_keeps_equal_timestamps_together(data):
    data.loc[128:, "date"] = data.loc[127, "date"]
    train, test = wf.split_data(data, "event", "Classification", date_column="date")
    assert train["date"].max() < test["date"].min()
    assert len(train) + len(test) == len(data)


@pytest.mark.parametrize("task,target", [("Regression", "regression"), ("Classification", "event")])
def test_model_artifacts_and_reload_prediction(data, tmp_path, task, target):
    data.loc[0, "x"] = np.nan
    result = wf.train_model(data, ["x", "segment"], target, task, wf.infer_types(data), tmp_path)
    assert all(path.is_file() for path in result.artifacts)
    assert result.metrics["test_rows"] == 32
    assert set(result.importance["variable"]) == {"x", "segment"}
    unseen = data.head(12).copy()
    unseen["segment"] = "never-seen-category"
    prediction, path = wf.predict_model(tmp_path / "model_train.pkl", unseen, tmp_path)
    assert len(prediction) == 12 and prediction["predicted"].notna().all()
    assert path.is_file()


def test_explicit_test_set_and_temporal_training(data, tmp_path):
    result = wf.train_model(data, ["x", "segment"], "regression", "Regression", wf.infer_types(data), tmp_path,
                            date_column="date")
    train, test = pd.read_csv(tmp_path / "train.csv"), pd.read_csv(tmp_path / "test.csv")
    assert train["date"].max() < test["date"].min()
    result = wf.train_model(data.iloc[:120], ["x"], "event", "Classification", wf.infer_types(data), tmp_path,
                            test_data=data.iloc[120:])
    assert result.metrics["test_rows"] == 40


@pytest.mark.parametrize("task,target", [("Regression", "regression"), ("Classification", "event")])
def test_binning_report_numeric_and_categorical(data, tmp_path, task, target):
    summary, paths = wf.variable_profile(data, ["x", "segment"], target, task, wf.infer_types(data), tmp_path)
    assert set(summary["name"]) == {"x", "segment"}
    assert all(path.is_file() for path in paths)
    assert "base64," in paths[-1].read_text(encoding="utf-8")
    assert set(pd.ExcelFile(paths[0]).sheet_names) == {"factor_1", "factor_2", "summary"}


def test_multiclass_binning_reports_unsupported_category(data, tmp_path):
    data["event"] = pd.cut(data["x"], bins=3, labels=False)
    summary, paths = wf.variable_profile(data, ["x", "segment"], "event", "Classification", wf.infer_types(data), tmp_path)
    assert "unsupported" in summary.set_index("name").loc["segment", "status"]
    assert paths[-1].is_file()


def test_profile_and_export(data, tmp_path):
    report = wf.data_profile(data.head(25), tmp_path)
    assert "<html" in report.read_text(encoding="utf-8").lower()
    for format in ["CSV", "HDF"]:
        path = wf.export_data(data, tmp_path, format)
        assert wf.load_data(path).shape == data.shape


def test_empty_and_invalid_target_are_actionable(data, tmp_path):
    with pytest.raises(ValueError, match="数据为空"):
        wf.normalize_data(pd.DataFrame())
    with pytest.raises(ValueError, match="特征变量"):
        wf.train_model(data, ["event"], "event", "Classification", wf.infer_types(data), tmp_path)


def test_marimo_app_loads_without_qt():
    from bibidataprofile.marimo_app import app
    outputs, definitions = app.run()
    assert len(outputs) > 10
    assert definitions["loaded_data"] is None


def test_marimo_demo_and_analysis_controls(tmp_path):
    from types import SimpleNamespace
    from bibidataprofile.marimo_app import app

    _, definitions = app.run(defs={"source_form": SimpleNamespace(value={"source": "示例数据", "output": str(tmp_path)})})
    assert len(definitions["loaded_data"]) == 300
    assert definitions["model_form"].value is None
