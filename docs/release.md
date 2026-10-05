# BibiDataProfile 发布说明

## 框架与源码

桌面框架位于 `desktop/`，来自 bibimapy 本地提交
`31b49bd96392bfc618ff32f9b50eb44a7973bb4c`，来源和 Apache-2.0 许可见
`desktop/UPSTREAM.md`、`desktop/LICENSE`。

`src/bibidataprofile/marimo_app.py` 是控制面板，
`src/bibidataprofile/workflows.py` 是独立分析服务。默认入口不加载 PyQt。
原始 PyQt 入口备份到 `legacy_main.py`；安装 `.[legacy]` 可用于历史代码维护。

## 用户操作

1. 导入表单选择示例、本地文件、上传文件或数据库 SQL，设置输出目录后提交。
2. 选择目标 Y、任务类型、特征 X，按需修改 JSON 类型并点击「应用变量设置」。
3. 在各标签页生成数据画像、变量分箱、训练模型、预测或导出数据。

连接使用 SQLAlchemy URL，例如 `sqlite:///D:/data/example.db`。
SQLite 开箱即用；其他数据库需要在应用 venv 内安装相应驱动。
SQL 只在提交导入表单时执行，不把数据库连接 URL 写入应用配置。
上传文件使用临时目录读取，完成后删除。

类型分配要求每列恰好属于一个类型。数值特征使用数值预处理，其他选择的特征
作为类别处理。分箱支持回归、二分类数值与类别特征、多分类数值特征；
多分类类别特征显示限制并保留重要性，不生成该特征的分箱图。
分类空目标行被删除；回归目标必须可转换为有限数值。
日期分割将最新的数据放入测试集，并保证相同时间戳不会跨越两侧。

模型将训练集拟合的预处理器与 LightGBM 一起保存；测试集和新数据复用该
预处理器。预测页面应选择由本应用生成且来源可信的模型文件。

## 本地构建与检查

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
python -m marimo check src/bibidataprofile/marimo_app.py
python scripts/smoke_server.py
cd desktop
npm ci
npm run application
npm run sidecar
npm run build
cargo fmt --manifest-path src-tauri/Cargo.toml -- --check
cargo test --manifest-path src-tauri/Cargo.toml
cargo clippy --manifest-path src-tauri/Cargo.toml --all-targets -- -D warnings
npm run tauri:build -- --bundles nsis
npm run portable
```

`prepare-application.mjs` 构建 wheel 并生成 `src-tauri/resources/app-wheel.rs`。
这些文件是构建产物，不提交。Python、npm、Tauri、Cargo 的版本号需要保持
一致。sidecar 可通过 `BIBIDATAPROFILE_UV` 指定 uv 路径。

## 运行配置

`~/.bibidataprofile/config.toml` 示例：

```toml
language = "system"
python_version = "3.12"
pip_index_url = "https://pypi.org/simple"
marimo_package = "marimo>=0.24.0,<0.25"
marimo_port = 2718
startup_timeout_seconds = 600
```

中文系统首次配置默认阿里云镜像。桌面设置可以修改镜像和桌面壳语言；
分析控制面板当前使用中文。关闭窗口时终止 marimo 子进程。
启动失败可查看 `~/.bibidataprofile/logs/marimo.log` 并点击重试。
便携 ZIP 内的 `bibidataprofile.exe` 和 `uv.exe` 必须保留在同一目录。

## GitHub 发布

仓库工作流 `.github/workflows/release.yml` 在语义版本标签触发后构建
Windows x86_64、macOS arm64/x86_64、Linux aarch64/x86_64。每个平台提供
三种包格式，文件名与 bibiocr 一致：Windows setup.exe/setup.msi/portable.zip，
macOS DMG/PKG/ZIP，Linux AppImage/RPM/tar.gz。macOS 构建前执行
`bash scripts/prepare-macos-runtime.sh`，把 OpenMP 库及许可一起放入应用。
Linux tar.gz 包含完整 AppDir，解压后执行 `AppRun`。
workflow_dispatch 构建并验证完整附件，不发布 Release。

提交并推送本次改动后，在已登录且具有仓库写权限的环境执行：

```powershell
git tag v0.2.2
git push bibiparrot v0.2.2
```

发布仓库为 `https://github.com/bibiparrot/bibidataprofile`。
流程先运行 Python 集成测试，每个架构再安装 wheel 运行测试，再构建桌面包。
最后聚合所有构建，校验 15 个桌面包和 wheel 是否齐全并生成 `SHA256SUMS.txt`，
全部成功才发布 Release，避免出现只有部分平台的公开版本。
无需将 Python 包预先发布到 PyPI。macOS、Linux 的完整安装运行需要在对应
平台验证；未配置 Windows 签名和 macOS notarization。
