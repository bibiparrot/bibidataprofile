# 0.2.3 验证记录

验证日期：2026-10-05。发布目标为 `bibiparrot/bibidataprofile`。

- Windows 上 15 项 Python 集成测试、marimo 静态检查通过。
- Playwright/Chromium 驱动真实控制面板，通过示例导入、变量设置、模型训练、
  分箱、数据画像、模型重载预测与 CSV 导出。报告和模型实际生成，五张界面截图
  保存在 `docs/screenshots/`；展示共用分析面板，不是桌面窗口装饰效果图。
- 界面增加设置生效提示，在变量设置提交前禁用训练按钮。
- Rust 格式与严格 Clippy、TypeScript/Vite 生产构建通过。
- CI 在五种原生架构上安装 wheel 运行分析测试，并检查包内容。最后强制校验
  15 个桌面包、wheel 和 SHA-256 完整性，全部成功后发布。
- macOS 附带 OpenMP，PKG 检查载荷、ZIP 检查运行文件；Linux tar.gz 保存
  完整 AppDir 和 AppRun，Windows portable ZIP 检查应用与 uv。

实际点击安装/卸载、macOS/Linux 图形桌面交互尚未验收，Windows 开发者签名
和 macOS 开发者签名/公证未配置。CI 通过状态以对应发布工作流为准。

## 0.2.1 历史记录

验证日期：2026-10-04，Windows x64。

- 15 项 Python 集成测试通过，覆盖导入、SQL、类型校验、日期分割、
  回归/分类训练、模型重载、二分类/多分类分箱、画像及数据导出。
- 全新 venv 从发布 wheel 安装后，同样的 15 项测试通过。
  marimo 0.24.2；未安装 PyQt5；已安装 wheel 的控制面板执行成功。
- marimo 静态检查通过。
- 发布依赖显式包含 setuptools 70–80，为 YData 提供 pkg_resources。
- Rust 3 项测试通过，最终代码的严格 Clippy 与格式检查通过。
- TypeScript/Vite 生产构建通过。
- marimo 在沙箱外启动并返回 HTTP 200；测试服务已停止。
- Windows NSIS 安装包和便携 ZIP 构建成功。ZIP 内程序与 release 程序
  字节一致，包含 uv、说明及 bibimapy 许可和来源信息。
- 发布 wheel 的 SHA-256 与 Rust 内嵌指纹一致。

未完成的验收：浏览器视觉交互、实际安装/卸载、macOS/Linux 安装运行。
应用内浏览器附加超时，Chrome 浏览器工具不可用，因此未宣称视觉验收通过。
