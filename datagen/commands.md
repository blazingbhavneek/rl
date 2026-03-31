# b0-b3 all at once, they are independent
python -m agents.b0_generate \
  --input-dir  ./functions/ \
  --output-dir ./sft/b0/ \
  --config     config.json \
  --mcp-tools-module my_mcp_tools \
  --concurrency 5 &

python -m agents.b1_generate \
  --input-dir  ./functions/ \
  --output-dir ./sft/b1/ \
  --config     config.json \
  --mcp-tools-module my_mcp_tools \
  --concurrency 5 &

python -m agents.b2_generate \
  --input-dir  ./functions/ \
  --output-dir ./sft/b2/ \
  --config     config.json \
  --mcp-tools-module my_mcp_tools \
  --concurrency 5 &

python -m agents.b3_generate \
  --input-dir  ./functions/ \
  --output-dir ./sft/b3/ \
  --config     config.json \
  --mcp-tools-module my_mcp_tools \
  --concurrency 5 &

wait

# b4: designer first, then writer
python -m agents.b4_task_designer \
  --input-dir          ./functions/ \
  --output-dir         ./sft/b4_tasks/ \
  --config             config.json \
  --mcp-tools-module   my_mcp_tools \
  --tasks-per-function 3 \
  --concurrency        3

python -m agents.b4_code_writer \
  --tasks-dir        ./sft/b4_tasks/ \
  --output-dir       ./sft/b4/ \
  --config           config.json \
  --mcp-tools-module my_mcp_tools \
  --concurrency      3

# Collect everything
python -m agents.collect_sft \
  --input-dirs ./sft/b0/ ./sft/b1/ ./sft/b2/ ./sft/b3/ ./sft/b4/ \
  --output     ./training/sft_pairs.jsonl
