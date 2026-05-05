@echo off
setlocal enabledelayedexpansion

title Anti-Covid — Startup

echo ============================================
echo  Anti-Covid AI - Stopping all processes...
echo ============================================

:: Clean ports (3000: Frontend, 8000: Backend)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":3000"') do (
    echo [INFO] Cleaning Port 3000, PID: %%a...
    taskkill /PID %%a /F >nul 2>&1
)

for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000"') do (
    echo [INFO] Cleaning Port 8000, PID: %%a...
    taskkill /PID %%a /F >nul 2>&1
)

:: Aggressive cleanup (optional)
taskkill /IM node.exe /F >nul 2>&1

echo [OK] Old processes stopped.
echo.

:: Define directories
set PROJECT_ROOT=%~dp0
set BACKEND_DIR=%PROJECT_ROOT%sabanci-main
set FRONTEND_DIR=%PROJECT_ROOT%frontend

echo ============================================
echo  Starting Backend (port 8000)...
echo ============================================

:: Backend venv check
if exist "%BACKEND_DIR%\.venv\Scripts\python.exe" (
    echo [INFO] Virtual environment found: .venv
    start "Anti-Covid Backend" cmd /k "cd /d %BACKEND_DIR% && .venv\Scripts\python.exe -m uvicorn backend.main:app --reload --port 8000"
) else (
    echo [WARNING] Virtual environment not found, trying system uvicorn.
    start "Anti-Covid Backend" cmd /k "cd /d %BACKEND_DIR% && uvicorn backend.main:app --reload --port 8000"
)

timeout /t 5 /nobreak >nul

echo ============================================
echo  Starting Frontend (port 3000)...
echo ============================================

if exist "%FRONTEND_DIR%\node_modules" (
    start "Anti-Covid Frontend" cmd /k "cd /d %FRONTEND_DIR% && npm run dev"
) else (
    echo [ERROR] Frontend node_modules not found!
    echo Please run 'cd frontend && npm install'.
    pause
    exit /b
)

echo.
echo ============================================
echo  System Started!
echo  Backend  : http://localhost:8000
echo  Frontend : http://localhost:3000
echo  Swagger  : http://localhost:8000/docs
echo ============================================
echo.
echo [INFO] Services will continue to run if you close this window.
echo [INFO] Use stop.bat to stop completely.
echo.
pause
