# BibiDataProfile 0.2.3

从数据导入到画像、变量分箱、模型训练与批量预测，在一个桌面工作台完成。

- 文件、上传与 SQL 数据导入，支持 CSV、TSV、JSON、XML、Excel、HDF。
- 数据预览、缺失值概览、HTML 数据画像，快速发现数据质量问题。
- 回归、二分类与多分类分析，输出 Excel 分箱表与可查看的分箱图。
- LightGBM 训练，随机、时间或独立测试集评估，保存模型后重复使用。
- 批量预测与数据导出，报告、指标和结果可直接下载。

本次发布位于 **bibiparrot/bibidataprofile**。提供 Windows x86_64、macOS arm64/x86_64、Linux aarch64/x86_64 的 15 种桌面包、Python wheel 和 SHA256SUMS.txt。

Windows：setup.exe / setup.msi / portable.zip。macOS：DMG / PKG / ZIP。Linux：AppImage / RPM / tar.gz。

首次启动需要联网下载 Python 3.12 和分析依赖；应用本身随安装包提供，之后可离线分析本地数据。macOS 包含 OpenMP 库。Windows 需要 WebView2；Linux 需要图形桌面和 OpenMP 运行库（libgomp1）。macOS 应用和安装包尚未配置开发者签名与公证。

截图、使用步骤和各平台直接下载入口见 [README](https://github.com/bibiparrot/bibidataprofile#readme)。
