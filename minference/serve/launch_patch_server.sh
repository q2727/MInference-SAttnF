#!/bin/bash

# 使用示例:
# bash minference/serve/launch_patch_server.sh Qwen/Qwen2.5-7B-Instruct-1M minference 8000

# 1. 获取项目根目录
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
LOG_DIR="$ROOT_DIR/minference/serve/logs"

MODEL_PATH=${1:-"meta-llama/Meta-Llama-3.1-8B-Instruct"}
PATCH_TYPE=${2:-"minference"}
PORT=${3:-8000}

echo "=================================================="
echo "Starting FastChat Patch Server"
echo "Root Dir:  $ROOT_DIR"
echo "Log Dir:   $LOG_DIR"
echo "Model:     $MODEL_PATH"
echo "Patch:     $PATCH_TYPE"
echo "=================================================="

# 准备日志目录
mkdir -p "$LOG_DIR"

# 设置 PYTHONPATH 确保能找到 minference 和 fastchat
export PYTHONPATH=$ROOT_DIR:$PYTHONPATH

# 切换工作目录到 log 目录启动进程
cd "$LOG_DIR"

# 1. 启动 Controller
echo "[1/3] Starting Controller..."
# 使用 tee 同时输出到屏幕和文件
python -u -m fastchat.serve.controller --host localhost --port 21001 2>&1 | tee controller.stdout.log & 
CONTROLLER_PID=$!
sleep 5

# 2. 启动 Patch Worker
echo "[2/3] Starting Patch Worker..."
python -u "$ROOT_DIR/minference/serve/patch_worker.py" \
    --model-path "$MODEL_PATH" \
    --enable-patch \
    --patch-type "$PATCH_TYPE" \
    --controller-address http://localhost:21001 \
    --port 21002 \
    --worker-address http://localhost:21002 2>&1 | tee worker.stdout.log & 
WORKER_PID=$!

# 3. 启动 OpenAI API Server
echo "[3/3] Starting OpenAI API Server..."
python -u -m fastchat.serve.openai_api_server --host localhost --port "$PORT" --controller-address http://localhost:21001 2>&1 | tee api_server.stdout.log & 
API_PID=$!

echo "--------------------------------------------------"
echo "Server startup initiated. Logs are streaming above."
echo "Server API: http://localhost:$PORT"
echo "Press Ctrl+C to stop all servers."
echo "--------------------------------------------------"

cd "$ROOT_DIR"

# 保持运行并捕捉退出信号
trap "kill $CONTROLLER_PID $WORKER_PID $API_PID; exit" SIGINT SIGTERM
wait