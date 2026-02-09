python simulation.py \
  --model_list gpt-4.1-mini gpt-4.1-mini gpt-4.1-mini  \
  --version v1 \
  --num_runs 1 \
  --example_folder example_generation/train \
  --simulation_folder example_trajectory/train \
  --search_mode \
  --appless \
  --instruct_agent_model unsloth/Qwen3-4B \
  --instruct_base_url http://localhost:8001/v1 \
  --example_ids 16 44 73 91 93 \
  --num_processes 5

python mixed_evaluation.py --example_folder example_trajectory/train/example_v1
