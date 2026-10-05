"""UI-independent data, reporting and model workflows for the marimo app."""

from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

DATA_TYPES = ("Numeric", "Categorical", "Text", "DateTime", "Id", "Unsupported")
FILE_TYPES = (".csv", ".tsv", ".json", ".xml", ".xlsx", ".xls", ".hdf", ".h5")


def normalize_data(data):
    if data is None or data.empty:
        raise ValueError("数据为空，请检查文件或 SQL 查询。")
    data = data.copy()
    data.columns = data.columns.map(str)
    if not data.columns.is_unique:
        raise ValueError("列名重复，请先为每一列设置唯一名称。")
    return data.reset_index(drop=True)


def load_data(path, *, encoding="auto", sheet=0):
    path = Path(path).expanduser()
    if not path.is_file():
        raise ValueError(f"找不到数据文件：{path}")
    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv", ".json", ".xml"):
        candidates = [encoding] if encoding != "auto" else ["utf-8-sig", "gb18030", "big5"]
        for candidate in candidates:
            try:
                if suffix in (".csv", ".tsv"):
                    data = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",", encoding=candidate)
                elif suffix == ".json":
                    data = pd.read_json(path, encoding=candidate)
                else:
                    data = pd.read_xml(path, encoding=candidate, parser="etree")
                return normalize_data(data)
            except UnicodeError:
                if candidate == candidates[-1]:
                    raise
    if suffix in (".xlsx", ".xls"):
        return normalize_data(pd.read_excel(path, sheet_name=sheet, engine="calamine"))
    if suffix in (".hdf", ".h5"):
        return normalize_data(pd.read_hdf(path))
    raise ValueError(f"不支持的文件格式：{suffix}")


def load_upload(name, contents, **kwargs):
    with tempfile.TemporaryDirectory(prefix="bibidataprofile-") as directory:
        path = Path(directory) / Path(name).name
        path.write_bytes(contents)
        return load_data(path, **kwargs)


def query_database(url, query):
    from sqlalchemy import create_engine, text

    if not query.strip():
        raise ValueError("请输入 SQL 查询。")
    engine = create_engine(url)
    try:
        with engine.connect() as connection:
            return normalize_data(pd.read_sql_query(text(query), connection))
    finally:
        engine.dispose()


def infer_types(data):
    result = {kind: [] for kind in DATA_TYPES}
    for name, series in data.items():
        unique = series.nunique()
        if unique == 0:
            kind = "Unsupported"
        elif pd.api.types.is_datetime64_any_dtype(series):
            kind = "DateTime"
        elif pd.api.types.is_bool_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
            kind = "Categorical"
        elif pd.api.types.is_numeric_dtype(series):
            kind = "Categorical" if unique <= 2 else "Numeric"
        elif unique <= max(20, len(series) * 0.05):
            kind = "Categorical"
        elif unique == len(series) and (name.lower() == "id" or name.lower().endswith("_id")):
            kind = "Id"
        else:
            kind = "Text"
        result[kind].append(name)
    return result


def parse_types(text, columns):
    mapping = json.loads(text)
    if not isinstance(mapping, dict) or set(mapping) - set(DATA_TYPES):
        raise ValueError(f"变量类型必须是 JSON 对象，键为 {', '.join(DATA_TYPES)}。")
    seen = []
    for kind, names in mapping.items():
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            raise ValueError(f"{kind} 必须是列名列表。")
        seen.extend(names)
    if len(seen) != len(set(seen)) or set(seen) != set(columns):
        raise ValueError("每一列必须恰好属于一种变量类型。")
    return {kind: mapping.get(kind, []) for kind in DATA_TYPES}


def output_directory(path):
    if not str(path).strip():
        raise ValueError("请输入输出目录。")
    directory = Path(path).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def export_data(data, directory, format="CSV"):
    directory = output_directory(directory)
    if format == "HDF":
        path = directory / "dataset.h5"
        data.to_hdf(path, key="data", mode="w", format="table")
    elif format == "CSV":
        path = directory / "dataset.csv"
        data.to_csv(path, index=False, encoding="utf-8-sig")
    else:
        raise ValueError("导出格式必须是 CSV 或 HDF。")
    return path


def data_profile(data, directory, *, minimal=True, title="BibiDataProfile 数据画像"):
    from ydata_profiling import ProfileReport

    path = output_directory(directory) / "data_profile.html"
    ProfileReport(data.copy(), title=title, minimal=minimal).to_file(path)
    return path


def validate_factors(data, features, target, task):
    if task not in ("Regression", "Classification"):
        raise ValueError("请选择回归或分类任务。")
    if target not in data or not features or target in features:
        raise ValueError("请选择目标变量和至少一个不同的特征变量。")
    if len(set(features)) != len(features) or set(features) - set(data.columns):
        raise ValueError("特征变量重复或不存在。")
    clean = data.loc[data[target].notna(), features + [target]].copy()
    if len(clean) < 10:
        raise ValueError("至少需要 10 行目标变量非空的数据。")
    if task == "Regression":
        clean[target] = pd.to_numeric(clean[target], errors="raise")
        if not np.isfinite(clean[target]).all():
            raise ValueError("回归目标包含无穷值。")
    elif clean[target].nunique() < 2:
        raise ValueError("分类目标至少需要两种类别。")
    return clean


def model_input(data, features, types):
    numeric = [name for name in features if name in types["Numeric"]]
    categorical = [name for name in features if name not in numeric]
    prepared = data[features].copy()
    for name in numeric:
        prepared[name] = pd.to_numeric(prepared[name], errors="raise").replace([np.inf, -np.inf], np.nan)
    for name in categorical:
        prepared[name] = prepared[name].map(lambda value: str(value) if pd.notna(value) else np.nan)
    return prepared, numeric, categorical


def model_pipeline(numeric, categorical, task):
    from lightgbm import LGBMClassifier, LGBMRegressor
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder

    preprocessing = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric),
        ("categorical", Pipeline([
            ("fill", SimpleImputer(strategy="constant", fill_value="缺失")),
            ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), categorical),
    ])
    estimator = LGBMRegressor if task == "Regression" else LGBMClassifier
    return Pipeline([
        ("preprocess", preprocessing),
        ("model", estimator(n_estimators=100, random_state=42, n_jobs=2, verbosity=-1)),
    ])


def split_data(data, target, task, ratio=0.2, date_column=None):
    from sklearn.model_selection import train_test_split

    if not 0 < ratio < 1:
        raise ValueError("测试集比例必须介于 0 和 1 之间。")
    if date_column:
        dates = pd.to_datetime(data[date_column], errors="raise")
        if dates.isna().any():
            raise ValueError("日期列包含空值。")
        ordered = data.loc[dates.sort_values(kind="stable").index]
        count = max(1, int(np.ceil(len(ordered) * ratio)))
        cutoff = dates.loc[ordered.index[-count]]
        train = ordered.loc[dates.loc[ordered.index] < cutoff]
        test = ordered.loc[dates.loc[ordered.index] >= cutoff]
        if train.empty or test.empty:
            raise ValueError("日期分割后训练集或测试集为空，请调整比例或日期列。")
        return train, test
    return train_test_split(data, test_size=ratio, random_state=42,
                            stratify=data[target] if task == "Classification" else None)


@dataclass
class ModelResult:
    metrics: dict
    importance: pd.DataFrame
    predictions: pd.DataFrame
    artifacts: list[Path]


def train_model(data, features, target, task, types, directory, *, ratio=0.2,
                date_column=None, test_data=None):
    import joblib
    from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score, root_mean_squared_error, roc_auc_score

    clean = validate_factors(data, features, target, task)
    if test_data is None:
        split_source = data.loc[clean.index].copy()
        split_source[target] = clean[target]
        train, test = split_data(split_source, target, task, ratio, date_column)
    else:
        train, test = clean, validate_factors(test_data, features, target, task)
    if task == "Classification" and train[target].nunique() < 2:
        raise ValueError("训练集至少需要两种类别。")
    train_x, numeric, categorical = model_input(train, features, types)
    test_x, _, _ = model_input(test, features, types)
    pipeline = model_pipeline(numeric, categorical, task)
    pipeline.fit(train_x, train[target])
    predicted = pipeline.predict(test_x)
    metrics = {"train_rows": len(train), "test_rows": len(test)}
    if task == "Regression":
        metrics.update(rmse=float(root_mean_squared_error(test[target], predicted)),
                       mae=float(mean_absolute_error(test[target], predicted)),
                       r2=float(r2_score(test[target], predicted)))
    else:
        metrics.update(accuracy=float(accuracy_score(test[target], predicted)),
                       f1_weighted=float(f1_score(test[target], predicted, average="weighted")))
        classes = pipeline.named_steps["model"].classes_
        if len(classes) == 2 and set(test[target].unique()) == set(classes):
            metrics["roc_auc"] = float(roc_auc_score(test[target] == classes[1], pipeline.predict_proba(test_x)[:, 1]))
    importance = pd.DataFrame({"variable": numeric + categorical,
                               "importance": pipeline.named_steps["model"].feature_importances_}).sort_values("importance", ascending=False)
    predictions = pd.DataFrame({"actual": test[target], "predicted": predicted}, index=test.index)
    directory = output_directory(directory)
    model_path = directory / "model_train.pkl"
    joblib.dump({"pipeline": pipeline, "features": features, "types": types, "target": target, "task": task}, model_path)
    train_path, test_path = directory / "train.csv", directory / "test.csv"
    train.to_csv(train_path, index=False, encoding="utf-8-sig")
    test.to_csv(test_path, index=False, encoding="utf-8-sig")
    predictions_path = directory / "predictions.csv"
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    metrics_path = directory / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = directory / "model_report.html"
    report_path.write_text("<meta charset='utf-8'><h1>模型评估</h1>" + pd.DataFrame([metrics]).to_html(index=False) + importance.to_html(index=False) + predictions.head(100).to_html(), encoding="utf-8")
    return ModelResult(metrics, importance, predictions, [model_path, train_path, test_path, predictions_path, metrics_path, report_path])


def predict_model(model_path, data, directory):
    import joblib

    model = joblib.load(Path(model_path).expanduser())
    prepared, _, _ = model_input(data, model["features"], model["types"])
    result = data.copy()
    name = "predicted"
    while name in result:
        name = "_" + name
    result[name] = model["pipeline"].predict(prepared)
    path = output_directory(directory) / "prediction_result.csv"
    result.to_csv(path, index=False, encoding="utf-8-sig")
    return result, path


def variable_profile(data, features, target, task, types, directory):
    import base64
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from optbinning import ContinuousOptimalBinning, MulticlassOptimalBinning, OptimalBinning
    from sklearn.preprocessing import LabelEncoder

    clean = validate_factors(data, features, target, task)
    prepared, numeric, categorical = model_input(clean, features, types)
    y = clean[target]
    if task == "Classification":
        y = LabelEncoder().fit_transform(y)
    model = model_pipeline(numeric, categorical, task)
    model.fit(prepared, y)
    importance = dict(zip(numeric + categorical, model.named_steps["model"].feature_importances_, strict=True))
    directory = output_directory(directory)
    rows, sections = [], []
    workbook = directory / "variable_profile.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        for index, name in enumerate(features):
            dtype = "numerical" if name in numeric else "categorical"
            if task == "Regression":
                binner = ContinuousOptimalBinning(name=name, dtype=dtype)
                metric = "mean"
            elif len(np.unique(y)) > 2:
                if dtype == "categorical":
                    rows.append({"name": name, "dtype": dtype, "status": "multiclass categorical unsupported",
                                 "feature_importance": importance[name], "missing": float(clean[name].isna().mean())})
                    sections.append(f"<h2>{escape(name)}</h2><p>多分类目标的类别特征仅计算特征重要性。</p>")
                    continue
                binner = MulticlassOptimalBinning(name=name)
                metric = "event_rate"
            else:
                binner = OptimalBinning(name=name, dtype=dtype)
                metric = "event_rate"
            binner.fit(prepared[name].to_numpy(), np.asarray(y))
            table = binner.binning_table
            frame = table.build()
            table.analysis(print_output=False)
            rows.append({"name": name, "dtype": dtype, "status": binner.status,
                         "n_bins": len(binner.splits) + 1 if dtype == "numerical" else len(binner.splits),
                         "quality_score": float(table.quality_score), "missing": float(clean[name].isna().mean()),
                         "feature_importance": importance[name]})
            frame.to_excel(writer, sheet_name=f"factor_{index + 1}", index=False)
            with tempfile.TemporaryDirectory() as temp:
                image_path = Path(temp) / "binning.png"
                try:
                    if isinstance(binner, MulticlassOptimalBinning):
                        table.plot(savefig=str(image_path))
                    else:
                        table.plot(metric=metric, savefig=str(image_path))
                    image = base64.b64encode(image_path.read_bytes()).decode("ascii")
                finally:
                    plt.close("all")
            sections.append(f"<h2>{escape(name)}</h2>{frame.to_html(index=False)}<img alt='binning' src='data:image/png;base64,{image}'>")
        summary = pd.DataFrame(rows).sort_values("feature_importance", ascending=False)
        summary.to_excel(writer, sheet_name="summary", index=False)
    report = directory / "binning_analysis.html"
    report.write_text("<!doctype html><meta charset='utf-8'><title>BibiDataProfile</title><h1>变量分箱分析</h1>" + summary.to_html(index=False) + "".join(sections), encoding="utf-8")
    return summary, [workbook, report]
