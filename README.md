# Manufacturing RL Scheduler

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/manufacturing-rl-scheduler/blob/main/src/main.py)

This project implements a reinforcement learning approach to optimize job shop scheduling problems, based on the methods described in *"Reinforcement Learning for Assignment Problem"* (Skomorokhov & Ovchinnikov, 2020). Our implementation extends the original research to manufacturing scheduling scenarios, focusing on minimizing makespan and operational penalties in dynamic environments.

## Research Basis

### Key Paper Contributions Implemented
- **Q-learning with Neural Networks**: Adapted the authors' Q-learning approach using deep neural networks as function approximators
- **Stochastic Environment Modeling**: Implemented dynamic job shop simulations with random machine breakdowns and priority changes
- **Penalty Minimization Strategy**: Developed reward shaping techniques to optimize for low penalty schedules
- **Greedy Baseline Comparison**: Included benchmark comparisons against traditional scheduling heuristics

```bibtex
@misc{skomorokhov2020reinforcement,
  title={Reinforcement Learning for Assignment Problem},
  author={Filipp Skomorokhov and George Ovchinnikov},
  year={2020},
  institution={Moscow Institute of Physics and Technology, Skolkovo Institute of Science and Technology}
}
```

**Demo Video:**

[![Dynamic Job Scheduling Demo](https://img.youtube.com/vi/placeholder/0.jpg)](src/DynamicJobScheduling.mp4)

*Note: The video will be displayed as a download link if embedded directly from the repository. For a proper preview, consider uploading the video to a platform like YouTube and replacing the placeholder link above with the actual video ID.*

## Features

* **RL-based Scheduling:** Uses deep Q-learning to train an agent that can make dynamic scheduling decisions.
* **Job Shop Simulation:** Simulates a realistic job shop environment with multiple machines and jobs with complex operation sequences.
* **Gantt Chart Visualization:** Provides interactive Gantt chart visualization of scheduling results using Plotly for clear analysis.
* **Modular Design:** The codebase is designed with modularity in mind, allowing for easy expansion and modification of agents, models, simulation, and training components.
* **Configurable Parameters:** Offers extensive configuration options for hyperparameters and problem instances to adapt to various scheduling scenarios.
```
## Project Structure
manufacturing-rl-scheduler/
├── env/
│   └── instances.txt           # Job shop problem instances data
├── src/
│   ├── agents.py               # Reinforcement learning agent implementation
│   ├── DynamicJobScheduling.mp4 # Example visualization video output
│   ├── __init__.py             # Initialization file for the source package
│   ├── main.py                 # Main execution script
│   ├── models.py               # Neural network model definitions
│   ├── simulator.py            # Job shop scheduling simulator
│   ├── train.py                # Agent training script
├── .gitignore                  # Git ignore file for version control
├── README.md                   # Project documentation (this file)
└── requirements.txt            # Python dependencies list

```

## Getting Started

### Prerequisites

* Python 3.6 or later
* pip (Python package installer)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/your-username/manufacturing-rl-scheduler.git](https://github.com/your-username/manufacturing-rl-scheduler.git)
    cd manufacturing-rl-scheduler
    ```

2.  **Create and activate a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate      # On Windows
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### Usage

1.  **Run the main script:**

    ```bash
    python src/main.py
    ```

    This will execute the agent training process and display the resulting schedule visualization as a Gantt chart.

2.  **Modify problem instances:**

    * The `env/instances.txt` file contains the problem instances. You can modify these to experiment with different scheduling scenarios.

3.  **Adjust hyperparameters:**

    * The `src/train.py` file contains various hyperparameters for training (e.g., learning rate, discount factor, exploration rate). Adjust these to optimize the training process.

4.  **View the demo video:**

    * The `src/DynamicJobScheduling.mp4` file shows an example visualization of the scheduling results.

## Dependencies

* torch
* numpy
* plotly
* anywidget

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please feel free to submit a pull request or open an issue.

## License

[MIT License](LICENSE) 

## Author

Tanvir Kabir Shaon


