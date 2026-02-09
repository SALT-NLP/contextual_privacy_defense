# [Contextualized Privacy Defense for LLM Agents](coming_soon)

**Authors:** Yule Wen, Yanzhe Zhang, Jianxun Lian, Xiaoyuan Yi, Xing Xie, Diyi Yang

## Abstract

Abstract LLM agents increasingly act on users’ personal information, yet existing privacy defenses remain limited in both design and adaptability. Most prior approaches rely on static or passive defenses, such as prompting and guarding. These paradigms are insufficient for supporting contextual, proactive privacy decisions in multi-step agent execution. We propose Contextualized Defense Instructing (CDI), a new privacy defense paradigm in which an instructor model generates step-specific, context-aware privacy guidance during execution, proactively shaping actions rather than merely constraining or vetoing them. Crucially, CDI is paired with an experience-driven optimization framework that trains the instructor via reinforcement learning (RL), where we convert failure trajectories with privacy violations into learning environments. We formalize baseline defenses and CDI as distinct intervention points in a canonical agent loop, and compare their privacy–helpfulness trade-offs within a unified simulation framework. Results show that our CDI consistently achieves a better balance between privacy preservation (94.2%) and helpfulness (80.6%) than baselines, with superior robustness to adversarial conditions and generalization.

## Architecture

### Core Components

1. **Agent System** (`agent.py`, `modules/`)
   - Implements the three-role simulation framework
   - defense families: prompting included in config files, guarding and CDI under `modules/`
 
2. **Simulation Engine** (`simulation.py`)
   - Simulate multi-agent communication

3. **Search Algorithm** (`search_control.py`)
   - Prompt optimization for attack and prompting defense

4. **Evaluation Framework** (`mixed_evaluation.py`)
   - Compute privacy protection rate (PP), helpfulness score (HS), and appropriate disclosure rate (AD)

5. **Training Framework** (`train.py`, `training_utils/`)
    - Trajectory selection standards
    - GRPO training loop for instructor model, guard model
    - Reward calculation

### Directory Structure

```
├── agent.py                  # Agent implementation
├── modules/                  # Defense implementation (guarding, CDI)
├── simulation.py             # Simulation
├── mixed_evaluation.py       # Evaluate simulation results
├── search_control.py         # Search algorithm
├── train.py                  # Training script for instructor and guard models
├── training_utils/           # Utilities for training
├── scripts/                  # bash scripts for running experiments
├── applications/             # Application-specific modules
├── example_generation/       # Training and test examples
└── camel/                    # A copy of the Camel framework, slightly modified
```

## Installation

### Prerequisites

### Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   Create a `.env` file in the root directory:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   # Optional: Add other model provider credentials, Google, Azure, etc
   ```

## Quick Start

### 1. Running Simulation with Vanilla Agents

Here is one command from the search process:

```bash
python simulation.py \
  --model_list gpt-4.1-mini gpt-4.1-mini gpt-4.1-mini  \
  --version v1 \
  --num_runs 1 \
  --example_folder example_generation/train \
  --simulation_folder example_trajectory/train \
  --search_mode \
  --appless \
  --example_ids 16 44 73 91 93 \
  --num_processes 5

python mixed_evaluation.py --example_folder example_trajectory/train/example_v1
```

**Parameters:**
- `--model_list`: Models for data_subject, data_sender, and data_recipient
- `--version`: Name of the example folder
- `--num_runs`: Number of simulations per configuration
- `--num_processes`: Number of parallel processes
- `--example_ids`: IDs of examples to simulate

### 2. Running Simulation with Privacy Defenses

To enable prompting, add defense prompts in the example generation config files. Or use the default prompts from PrivacyLens with `--prompting`.

To enable guarding, add `--guard_agent_model <model_name>` and `--guard_base_url <base_url>` to specify the guard model.

To enable CDI, add `--instruct_agent_model <model_name>` and `--instruct_base_url <base_url>` to specify the instructor model.

### 3. Running Adversarial Attack

```bash
python search_control.py \
   --config_dir  example_generation/train \
   --output_dir example_generation/train_attacked \
   --example_ids 16 \
   --search_dir_list train_attacked_16 \
   --attack_num_examples 5 \
   --num_tasks 10 \
   --num_runs 1 \
   --goal attack \
   --appless \
   --num_processes 10 \
   --max_simulation_round 2 \
   --keep_bank \
   --target_exp_result 0.0 \
   --adaptive_search \
   --prompt_version v1 \
   --no_backtrack \
   --instruct_agent_model unsloth/Qwen3-4B \
   --instruct_base_url http://localhost:8001/v1 \
   --data_sender_model gpt-4.1-mini \
   --data_subject_model gpt-4.1-mini \
   --data_recipient_model gpt-4.1-mini \
   --search_agent_model gpt-5
```

**Key Parameters:**
- `--attack_num_examples`: Number of trajectories for reflection, use `--defense_num_examples` for defense
- `--num_tasks`: Number of threads (default in paper: 30), not used for defense
- `--num_runs`: Number of simulations per configuration (default in paper: 1)
- `--num_processes`: Number of parallel process used for experiments
- `--max_simulation_round`: Total search steps (default in paper: 10)


### 4. Prompting Optimization via Search

```bash
python search_control.py \
   --config_dir  example_generation/train \
   --output_dir example_generation/train_prompting_5 \
   --example_ids 16 44 73 91 93 \
   --search_dir_list train_prompting_5 \
   --defense_num_examples 5 \
   --num_tasks 1 \
   --num_runs 2 \
   --goal defense \
   --appless \
   --num_processes 10 \
   --max_simulation_round 2 \
   --keep_bank \
   --target_exp_result 1.0 \
   --adaptive_search \
   --prompt_version v1 \
   --no_backtrack \
   --use_mixed_objective \
   --data_sender_model gpt-4.1-mini \
   --data_subject_model gpt-4.1-mini \
   --data_recipient_model gpt-4.1-mini \
   --search_agent_model gpt-5
```

**Key Parameters:**
- `--defense_num_examples`: Number of trajectories for reflection, use `--attack_num_examples` for attack
- `--use_mixed_objective`: Whether to use AD for optimization, default use PP
- `--num_runs`: Number of simulations per configuration (default in paper: 10)
- `--max_simulation_round`: Total search steps (default in paper: 10)

### 5. Training Instructor and Guard Models

```bash
python train.py --config scripts/training.yaml
```

You can modify the training configurations in `scripts/training.yaml`.

**Key Parameters:**
- `model_name_or_path`: Base model for instructor or guard
- `dataset_instruct_folder`: Folder containing training examples for instructor model
- `dataset_guard_folder`: Folder containing training examples for guard model
- `use_mixed_objective`: Whether to use AD for training instructor model, default use PP


## Acknowledgments

- Built on the [Search for Privacy Risks](https://github.com/SALT-NLP/search_privacy_risk.git) for simulation, attack and optimization for prompting method
- Use [CAMEL framework](https://github.com/camel-ai/camel) for multi-agent systems
- Use [LiteLLM](https://github.com/BerriAI/litellm) for unified model access
- Use [Open-R1] (https://github.com/huggingface/open-r1.git) for GRPO training framework
