bibidataprofile portable for Windows x64
==================================

Keep bibidataprofile.exe and uv.exe in the same directory, then run bibidataprofile.exe.
No installer or system Python is required. On first launch, uv downloads the
configured Python 3.12 runtime, marimo, and its Python dependencies into:

    %USERPROFILE%\.bibidataprofile

Microsoft Edge WebView2 Runtime is required. It is included with current
Windows 10 and Windows 11 installations and can also be installed from
Microsoft if it is missing.

Configuration:

    %USERPROFILE%\.bibidataprofile\config.toml

The local marimo service listens only on 127.0.0.1 and stops when bibidataprofile
exits.
