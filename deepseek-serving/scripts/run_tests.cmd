@echo off
rem 跑全部测试（优先带 torch 的 py -3.12；不行就用默认 python）
cd /d %~dp0..
for %%f in (tests\test_*.py) do (
  echo ===== %%f =====
  py -3.12 %%f 2>nul || python %%f
)
