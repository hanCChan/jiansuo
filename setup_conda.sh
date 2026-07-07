#!/usr/bin/env bash
# 激活项目本地 Miniconda（无需写 ~/.bashrc）
export PATH="/data1/hcc/jiansuo/miniconda3/bin:$PATH"
# 初始化 conda 到当前 shell
eval "$(/data1/hcc/jiansuo/miniconda3/bin/conda shell.bash hook)"
conda activate jiansuo-embed
echo "conda: $(conda --version)"
echo "python: $(python --version)"
echo "env: $CONDA_DEFAULT_ENV"
