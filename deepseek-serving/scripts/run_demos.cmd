@echo off
cd /d %~dp0..
echo ===== PD demo =====
py -3.12 examples\demo_pd.py 2>nul || python examples\demo_pd.py
echo ===== EP demo =====
py -3.12 examples\demo_ep.py 2>nul || python examples\demo_ep.py
