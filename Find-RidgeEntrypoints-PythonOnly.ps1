# Run from repo root
$patterns = "run_train_ridge|train_ridge|build_next_season_dataset|feat_points|weekly_to_season"

Get-ChildItem -Path src, tests -Recurse -File -Filter *.py |
    Where-Object { $_.FullName -notmatch "\\__pycache__\\" } |
    Select-String -Pattern $patterns |
    ForEach-Object {
        "{0}:{1}:{2}" -f $_.Path, $_.LineNumber, $_.Line.Trim()
    }