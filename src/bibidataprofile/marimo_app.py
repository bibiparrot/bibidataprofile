import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full", app_title="BibiDataProfile")


@app.cell
def _():
    import json
    from pathlib import Path
    import marimo as mo
    import numpy as np
    import pandas as pd
    from bibidataprofile import workflows as wf

    def artifact_links(paths):
        return mo.hstack([
            mo.download(path.read_bytes(), filename=path.name, label=f"下载 {path.name}")
            for path in paths if path.is_file()
        ], wrap=True, justify="start")

    def failure(error):
        return mo.callout(f"操作失败：{error}", kind="danger")

    def control_form(fields, *, submit_button_label, submit_button_disabled=False, advanced=(), advanced_label="文件与数据库选项"):
        def field(key):
            return '<div class="bdp-field" style="min-width:0">{' + key + '}</div>'
        primary = "".join(field(key) for key in fields if key not in advanced)
        extra = "".join(field(key) for key in advanced)
        template = '<div class="bdp-grid" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px;padding:16px 0;font-family:system-ui,sans-serif">' + primary + '</div>'
        if extra:
            template += '<details class="bdp-advanced" style="border:1px solid #cadde2;border-radius:10px;padding:14px 18px;margin-bottom:12px"><summary style="cursor:pointer;color:#176e7a;font-weight:600">' + advanced_label + '</summary><div style="display:grid;gap:16px;padding-top:16px">' + extra + '</div></details>'
        return mo.Html(template).batch(**fields).form(
            submit_button_label=submit_button_label,
            submit_button_disabled=submit_button_disabled,
            bordered=False,
        )

    return Path, artifact_links, control_form, failure, json, mo, np, pd, wf


@app.cell
def _(mo):
    mo.vstack([
        mo.Html("""<style>
        .bdp-hero {padding:28px 32px;border-radius:18px;background:linear-gradient(115deg,#103d52,#176e7a);color:#fff;margin-bottom:18px;font-family:system-ui,sans-serif}
        .bdp-hero h1 {margin:6px 0 12px;font-size:36px;color:#fff;font-weight:700}
        .bdp-hero p {margin:0;font-size:16px;color:#d7edf0;line-height:1.6}
        .bdp-eyebrow {font-size:12px;letter-spacing:2px;color:#9adce0}
        .bdp-grid {display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px;padding:20px 0;font-family:system-ui,sans-serif}
        .bdp-field {min-width:0}
        .bdp-field > * {width:100%}
        .bdp-advanced {border:1px solid #cadde2;border-radius:10px;padding:14px 18px;margin-bottom:12px;font-family:system-ui,sans-serif}
        .bdp-advanced summary {cursor:pointer;color:#176e7a;font-weight:600}
        @media(max-width:760px) {.bdp-grid {grid-template-columns:1fr}.bdp-hero {padding:22px}}
        </style><div class="bdp-hero"><span class="bdp-eyebrow">从数据到洞察</span>
        <h1>BibiDataProfile</h1><p>发现数据问题，理解关键变量，构建可复用的预测模型。</p>
        <p>数据导入 · 数据画像 · 变量分箱 · 模型训练 · 批量预测</p></div>"""),
        mo.md("选择「示例数据」即可开始体验，也可以导入自己的文件或数据库查询结果。"),
    ])
    return


@app.cell
def _(Path, control_form, mo, wf):
    source_form = control_form({
        "source": mo.ui.dropdown(["示例数据", "本地文件", "上传文件", "数据库 SQL"], value="示例数据", label="数据来源"),
        "path": mo.ui.text(placeholder=r"D:\data\dataset.csv", label="本地文件路径", full_width=True),
        "upload": mo.ui.file(filetypes=list(wf.FILE_TYPES), label="选择上传文件"),
        "encoding": mo.ui.dropdown(["auto", "utf-8-sig", "gb18030", "big5"], value="auto", label="文本编码"),
        "sheet": mo.ui.text(placeholder="留空使用第一个工作表", label="Excel 工作表名称"),
        "url": mo.ui.text(placeholder="sqlite:///D:/data/example.db", label="SQLAlchemy 数据库连接 URL", kind="password", full_width=True),
        "query": mo.ui.code_editor(value="SELECT * FROM your_table", language="sql", label="SQL 查询", min_height=110),
        "output": mo.ui.text(value=str(Path.home() / ".bibidataprofile" / "reports"), label="报告及模型输出目录", full_width=True),
    }, submit_button_label="导入数据", advanced=("path", "upload", "encoding", "sheet", "url", "query"))
    source_form
    return (source_form,)


@app.cell
def _(failure, mo, np, pd, source_form, wf):
    loaded_data, load_error, report_directory = None, None, None
    if source_form.value is not None:
        _config = source_form.value
        report_directory = _config["output"]
        try:
            if _config["source"] == "示例数据":
                _rng = np.random.default_rng(42)
                _x = _rng.normal(size=300)
                loaded_data = pd.DataFrame({
                    "amount": _x, "visits": _rng.integers(1, 30, 300),
                    "segment": _rng.choice(["A", "B", "C"], 300),
                    "event": (_x + _rng.normal(size=300) * 0.4 > 0).astype(int),
                    "value": _x * 10 + _rng.normal(size=300),
                    "date": pd.date_range("2025-01-01", periods=300),
                })
            elif _config["source"] == "数据库 SQL":
                loaded_data = wf.query_database(_config["url"], _config["query"])
            elif _config["source"] == "上传文件":
                if not _config["upload"]:
                    raise ValueError("请先选择上传文件。")
                _file = _config["upload"][0]
                loaded_data = wf.load_upload(_file.name, _file.contents, encoding=_config["encoding"], sheet=_config["sheet"] or 0)
            else:
                loaded_data = wf.load_data(_config["path"], encoding=_config["encoding"], sheet=_config["sheet"] or 0)
        except Exception as _error:
            load_error = failure(_error)
    mo.output.replace(load_error if load_error else mo.md(""))
    return loaded_data, report_directory


@app.cell
def _(loaded_data, mo):
    if loaded_data is not None:
        mo.output.replace(mo.vstack([
            mo.hstack([
                mo.stat(len(loaded_data), label="数据行数", bordered=True),
                mo.stat(len(loaded_data.columns), label="变量数", bordered=True),
                mo.stat(int(loaded_data.isna().sum().sum()), label="缺失值", bordered=True),
            ]),
            mo.accordion({"数据预览": mo.ui.table(loaded_data.head(1000), selection=None, page_size=10)}),
        ]))
    return


@app.cell
def _(loaded_data, mo):
    _columns = list(loaded_data.columns) if loaded_data is not None else ["请先导入数据"]
    target_control = mo.ui.dropdown(_columns, value=_columns[0], label="目标变量 Y", searchable=True)
    return (target_control,)


@app.cell
def _(control_form, json, loaded_data, mo, target_control, wf):
    inferred_types = wf.infer_types(loaded_data) if loaded_data is not None else {kind: [] for kind in wf.DATA_TYPES}
    _options = [] if loaded_data is None else [name for name in loaded_data.columns if name != target_control.value]
    _default = [name for name in _options if name in inferred_types["Numeric"] + inferred_types["Categorical"]]
    factor_form = control_form({
        "task": mo.ui.dropdown({"回归": "Regression", "分类": "Classification"}, value="回归", label="任务类型"),
        "features": mo.ui.multiselect(_options, value=_default, label="特征变量 X"),
        "types": mo.ui.code_editor(value=json.dumps(inferred_types, ensure_ascii=False, indent=2), language="json", label="变量类型（可修改）", min_height=200),
    }, submit_button_label="应用变量设置", advanced=("types",), advanced_label="变量类型设置（自动识别，可手动调整）")
    return factor_form, inferred_types


@app.cell
def _(control_form, loaded_data, mo):
    profile_form = control_form({
        "minimal": mo.ui.checkbox(value=True, label="快速画像（关闭后生成完整报告）"),
        "title": mo.ui.text(value="BibiDataProfile 数据画像", label="报告标题"),
    }, submit_button_label="生成数据画像", submit_button_disabled=loaded_data is None)
    variable_button = mo.ui.run_button(label="生成变量分箱报告", disabled=loaded_data is None)
    export_form = mo.ui.dropdown(["CSV", "HDF"], value="CSV", label="导出数据格式").form(submit_button_label="导出当前数据", submit_button_disabled=loaded_data is None)
    return export_form, profile_form, variable_button


@app.cell
def _(control_form, factor_form, loaded_data, mo, target_control):
    _dates = ["随机分割"] + (list(loaded_data.columns) if loaded_data is not None else [])
    model_form = control_form({
        "ratio": mo.ui.slider(0.05, 0.5, step=0.05, value=0.2, label="测试集比例"),
        "date": mo.ui.dropdown(_dates, value="随机分割", label="按日期列分割（最新数据用于测试）"),
        "test_path": mo.ui.text(placeholder="留空自动分割；填写时当前数据用作训练集", label="独立测试集路径", full_width=True),
    }, submit_button_label="训练 LightGBM 模型", submit_button_disabled=loaded_data is None or factor_form.value is None)
    prediction_form = control_form({
        "model_path": mo.ui.text(placeholder=".../model_train.pkl", label="已保存的模型路径", full_width=True),
        "data_path": mo.ui.text(placeholder="留空使用当前数据", label="预测数据文件路径", full_width=True),
    }, submit_button_label="预测并导出", submit_button_disabled=loaded_data is None)
    return model_form, prediction_form


@app.cell
def _(artifact_links, failure, loaded_data, mo, profile_form, report_directory, wf):
    profile_result = mo.md("生成报告后可在这里预览和下载。")
    if loaded_data is not None and profile_form.value is not None:
        try:
            with mo.status.spinner(title="正在生成数据画像…"):
                _path = wf.data_profile(loaded_data, report_directory, **profile_form.value)
            profile_result = mo.vstack([artifact_links([_path]), mo.iframe(_path.read_text(encoding="utf-8"), height="700px")])
        except Exception as _error:
            profile_result = failure(_error)
    return (profile_result,)


@app.cell
def _(artifact_links, export_form, failure, loaded_data, mo, report_directory, wf):
    export_result = mo.md("")
    if loaded_data is not None and export_form.value is not None:
        try:
            _path = wf.export_data(loaded_data, report_directory, export_form.value)
            export_result = mo.vstack([mo.md(f"已保存到 `{_path}`"), artifact_links([_path])])
        except Exception as _error:
            export_result = failure(_error)
    return (export_result,)


@app.cell
def _(artifact_links, factor_form, failure, loaded_data, mo, report_directory, target_control, variable_button, wf):
    variable_result = mo.md("先应用变量设置，再生成分箱报告。多分类的类别特征仅计算重要性。")
    if variable_button.value and loaded_data is not None:
        try:
            if factor_form.value is None:
                raise ValueError("请先点击「应用变量设置」。")
            _settings = factor_form.value
            _types = wf.parse_types(_settings["types"], loaded_data.columns)
            with mo.status.spinner(title="正在分析变量和分箱…"):
                _summary, _paths = wf.variable_profile(loaded_data, _settings["features"], target_control.value,
                                                       _settings["task"], _types, report_directory)
            variable_result = mo.vstack([mo.ui.table(_summary, selection=None), artifact_links(_paths),
                                          mo.iframe(_paths[-1].read_text(encoding="utf-8"), height="650px")])
        except Exception as _error:
            variable_result = failure(_error)
    return (variable_result,)


@app.cell
def _(artifact_links, factor_form, failure, loaded_data, mo, model_form, report_directory, target_control, wf):
    model_result = mo.md("应用变量设置后，可训练模型并导出训练集、测试集、评估指标和预测结果。")
    if loaded_data is not None and model_form.value is not None:
        try:
            if factor_form.value is None:
                raise ValueError("请先点击「应用变量设置」。")
            _settings, _split = factor_form.value, model_form.value
            _types = wf.parse_types(_settings["types"], loaded_data.columns)
            _test = wf.load_data(_split["test_path"]) if _split["test_path"].strip() else None
            with mo.status.spinner(title="正在训练与评估模型…"):
                _result = wf.train_model(loaded_data, _settings["features"], target_control.value, _settings["task"],
                                         _types, report_directory, ratio=_split["ratio"],
                                         date_column=None if _split["date"] == "随机分割" else _split["date"], test_data=_test)
            _labels = {"train_rows": "训练样本", "test_rows": "测试样本", "rmse": "RMSE", "mae": "MAE", "r2": "R²", "accuracy": "准确率", "f1": "F1", "roc_auc": "ROC AUC"}
            _scores = mo.hstack([mo.stat(round(_score, 4) if isinstance(_score, float) else _score,
                                        label=_labels.get(_key, _key), bordered=True)
                                for _key, _score in _result.metrics.items()], wrap=True)
            model_result = mo.vstack([_scores, mo.ui.table(_result.importance, selection=None),
                                      mo.ui.table(_result.predictions.head(100), selection=None), artifact_links(_result.artifacts)])
        except Exception as _error:
            model_result = failure(_error)
    return (model_result,)


@app.cell
def _(artifact_links, failure, loaded_data, mo, prediction_form, report_directory, wf):
    prediction_result = mo.md("选择由本应用保存的模型。预测数据可以包含训练时未出现的类别。")
    if loaded_data is not None and prediction_form.value is not None:
        try:
            _settings = prediction_form.value
            _data = wf.load_data(_settings["data_path"]) if _settings["data_path"].strip() else loaded_data
            with mo.status.spinner(title="正在预测…"):
                _result, _path = wf.predict_model(_settings["model_path"], _data, report_directory)
            prediction_result = mo.vstack([mo.ui.table(_result.head(100), selection=None), artifact_links([_path])])
        except Exception as _error:
            prediction_result = failure(_error)
    return (prediction_result,)


@app.cell
def _(export_form, export_result, factor_form, loaded_data, mo, model_form, model_result, prediction_form, prediction_result, profile_form, profile_result, target_control, variable_button, variable_result):
    if loaded_data is not None:
        mo.output.replace(mo.vstack([
            mo.vstack([mo.md("## 目标与变量设置"), target_control, factor_form,
                       mo.callout(f"已应用变量设置：{target_control.value} · {len(factor_form.value['features'])} 个特征", kind="success")
                       if factor_form.value is not None else mo.md("选择目标和特征后，点击「应用变量设置」继续。")]),
            mo.ui.tabs({
                "数据画像": mo.vstack([profile_form, profile_result]),
                "变量分箱": mo.vstack([variable_button, variable_result]),
                "模型构建": mo.vstack([model_form, model_result]),
                "模型预测": mo.vstack([prediction_form, prediction_result]),
                "数据导出": mo.vstack([export_form, export_result]),
            }),
        ]))
    return


if __name__ == "__main__":
    app.run()
