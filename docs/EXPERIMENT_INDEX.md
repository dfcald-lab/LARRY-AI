# Experiment Index

This index links to the numbered experiments in LARRY-AI. The sequence records the development of the project, from basic neural-network components to experiments on representation and optimization.

For detailed methods, results, limitations, and conclusions, see the [Project Journal](PROJECT_JOURNAL.md).

**Total: 77 numbered experiments.**

## Phase 1 — Neural-Network Foundations

Build the basic neurons, layers, networks, training methods, activation functions, and initialization experiments.

- [001 — First Learning System](../experiments/001_first_learning_system/)
- [002 — Tiny Neuron](../experiments/002_tiny_neuron/)
- [003 — Gradient Descent](../experiments/003_gradient_descent/)
- [004 — Learning Rate](../experiments/004_learning_rate/)
- [005 — Multiple Inputs](../experiments/005_multiple_inputs/)
- [006 — Two Neuron Layer](../experiments/006_two_neuron_layer/)
- [007 — Activation Function](../experiments/007_activation_function/)
- [008 — Trainable ReLU](../experiments/008_trainable_relu/)
- [009 — ReLU Gradient](../experiments/009_relu_gradient/)
- [010 — First Multilayer Network](../experiments/010_first_multilayer_network/)
- [011 — Reusable Neuron](../experiments/011_reusable_neuron/)
- [012 — Reusable Layer](../experiments/012_reusable_layer/)
- [013 — Automatic Layer](../experiments/013_automatic_layer/)
- [014 — Training Loop](../experiments/014_training_loop/)
- [015 — Trainable Layer](../experiments/015_trainable_layer/)
- [016 — Multilayer Network](../experiments/016_multilayer_network/)
- [017 — Reusable Network](../experiments/017_reusable_network/)
- [018 — Automatic Network](../experiments/018_automatic_network/)
- [019 — Initialization Test](../experiments/019_initialization_test/)
- [020 — Initialization Scale](../experiments/020_initialization_scale/)
- [021 — He Initialization](../experiments/021_he_initialization/)
- [022 — Activation Gradient Measurements](../experiments/022_activation_gradient_measurements/)

## Phase 2 — Representation and Training Dynamics

Investigate activation patterns, representation geometry, training order, learning-rate stability, gradient conflict, and hidden-neuron death.

- [023 — Training Activation Checkpoints](../experiments/023_training_activation_checkpoints/)
- [024 — Neuron Lifetime](../experiments/024_neuron_lifetime/)
- [025 — Hidden Representation](../experiments/025_hidden_representation/)
- [026 — Representation Geometry](../experiments/026_representation_geometry/)
- [027 — Separability Transitions](../experiments/027_separability_transitions/)
- [028 — Transition Microscope](../experiments/028_transition_microscope/)
- [029 — Gradient Geometry](../experiments/029_gradient_geometry/)
- [030 — Training Order](../experiments/030_training_order/)
- [031 — Online vs. Batch](../experiments/031_online_vs_batch/)
- [032 — Batch Gradient Scaling](../experiments/032_batch_gradient_scaling/)
- [033 — Learning Rate Sweep](../experiments/033_learning_rate_sweep/)
- [034 — Early Trajectory Discriminator](../experiments/034_early_trajectory_discriminator/)
- [035 — Hidden Neuron Death Guard](../experiments/035_hidden_neuron_death_guard/)
- [036 — Gradient Conflict](../experiments/036_gradient_conflict/)
- [037 — Gradient Floor Rescue](../experiments/037_gradient_floor_rescue/)

## Phase 3 — Hidden Capacity and Rescue Features

Study hidden capacity, feature injection, rescue initialization, and how new features contribute to task outputs.

- [038 — Hidden Capacity Sweep](../experiments/038_hidden_capacity_sweep/)
- [039 — Effective Capacity Pruning](../experiments/039_effective_capacity_pruning/)
- [040 — Late Capacity Injection](../experiments/040_late_capacity_injection/)
- [041 — Minimum Rescue Capacity](../experiments/041_minimum_rescue_capacity/)
- [042 — Capacity Injection Timing](../experiments/042_capacity_injection_timing/)
- [043 — Rescue Initialization Sensitivity](../experiments/043_rescue_initialization_sensitivity/)
- [044 — Rescue Activation Coverage](../experiments/044_rescue_activation_coverage/)
- [045 — Rescue Pattern Analysis](../experiments/045_rescue_pattern_analysis/)
- [046 — Rescue Early Dynamics](../experiments/046_rescue_early_dynamics/)
- [047 — Nonzero Rescue Output Weights](../experiments/047_nonzero_rescue_output_weights/)
- [048 — Signed Rescue Output Weights](../experiments/048_signed_rescue_output_weights/)
- [049 — Balanced Rescue Output Weights](../experiments/049_balanced_rescue_output_weights/)
- [050 — Learned Rescue Contributions](../experiments/050_learned_rescue_contributions/)
- [051 — Rescue Contribution Cap](../experiments/051_rescue_contribution_cap/)
- [052 — Task Contribution Cap](../experiments/052_task_contribution_cap/)
- [053 — Rescue Contribution Boost](../experiments/053_rescue_contribution_boost/)

## Phase 4 — Feature Structure and Learning

Investigate feature update scaling and direction, task alignment, complementarity, overlap, feature strength, and training budgets.

- [054 — Hidden Feature Update Scaling](../experiments/054_hidden_feature_update_scaling/)
- [055 — Hidden Feature Direction](../experiments/055_hidden_feature_direction/)
- [056 — Task Aligned Features](../experiments/056_task_aligned_features/)
- [057 — Hidden Feature Complementarity](../experiments/057_hidden_feature_complementarity/)
- [058 — Complementary Initialization](../experiments/058_complementary_initialization/)
- [059 — Input Space Complementarity](../experiments/059_input_space_complementarity/)
- [060 — Complementarity Generalization](../experiments/060_complementarity_generalization/)
- [061 — Feature Overlap Sweep](../experiments/061_feature_overlap_sweep/)
- [062 — Task Coverage Matrix](../experiments/062_task_coverage_matrix/)
- [063 — Feature Strength Sweep](../experiments/063_feature_strength_sweep/)
- [064 — Feature Strength Convergence](../experiments/064_feature_strength_convergence/)
- [065 — Strength Training Budget](../experiments/065_strength_training_budget/)
- [066 — Structure Strength Budget](../experiments/066_structure_strength_budget/)

## Phase 5 — Representation, Optimization, and Gradients

Examine representation stability, output initialization and coupling, hidden-layer gradients, gradient decomposition, and training trajectories.

- [067 — Representation vs. Optimization](../experiments/067_representation_vs_optimization/)
- [068 — Representation Trajectory](../experiments/068_representation_trajectory/)
- [069 — Frozen Representation](../experiments/069_frozen_representation/)
- [070 — Representation Stability](../experiments/070_representation_stability/)
- [071 — Output Initialization](../experiments/071_output_initialization/)
- [072 — New Feature Output Coupling](../experiments/072_new_feature_output_coupling/)
- [073 — Gradient Direction](../experiments/073_gradient_direction/)
- [074 — Hidden Gradient Measurement](../experiments/074_hidden_gradient_measurement/)
- [075 — Gradient Decomposition](../experiments/075_gradient_decomposition/)
- [076 — Gradient To Trajectory](../experiments/076_gradient_to_trajectory/)
- [077 — Early Warning Signals](../experiments/077_early_warning_signals/) — Tests whether early loss changes, parameter movement, and ReLU activation patterns are associated with later instability across 320 runs. The relationships vary by output coupling; no universal predictor was established.

## Language Model Foundations

Begin building the components Larry needs to learn from text.

- [078 — Character Tokenization](../experiments/078_character_tokenization/) — Converts characters into deterministic integer IDs, reconstructs known text, and handles unknown characters. All five tests passed.
- [079 — Character Bigram Model](../experiments/079_character_bigram_model/) — Learns next-character probabilities from adjacent-character counts. Achieved 49.5% training accuracy versus 13.3% for the most-common-character baseline; generalization remains untested.

## How to Use This Index

Open an experiment folder to inspect its script and any supporting files. Read the corresponding journal section for the research question, experimental setup, results, and interpretation.

The folder names provide the experiment sequence. The journal provides the detailed record of what was tested and what the evidence supports.
