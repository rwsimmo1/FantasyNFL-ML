# Run from repo root
Get-ChildItem -Path src, tests -Recurse -File |
    Sort-Object FullName |
    ForEach-Object { $_.FullName }