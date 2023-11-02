# MultiAgentSurveillanceAnomalyDetection
Multi-Agent Robot Learning algorithm using Deep Active Inference (DAI) for anomaly detection and Soft Actor Critic decomposed for multi-agent settings (mSAC) tested in the CARLA autonomous driving simulator and the StreetHazards data set to train and test the DAI algorithm

# Multi-Agent Surveillance and Anomaly Detection in Autonomous Vehicle Environments using Deep Active Inference and Soft Actor-Critic
# I. INTRODUCTION
Ajith Senthil
architectures perform in robot learning settings
In the landscape of autonomous systems, surveillance stands as a critical application, especially in dynamic urban environ- ments. Ensuring real-time anomaly detection is paramount for effective surveillance. This project seeks to leverage the power of Deep Active Inference (DAI) [3] for anomaly detection combined with Soft Actor-Critic (mSAC) decomposed for multi-agent settings [5] for efficient multi-agent collabora- tion in the CARLA simulator environment [1], emphasizing surveillance scenarios.
# II. PROBLEM STATEMENT
The primary objective of this project is to develop a system that can efficiently detect anomalies in autonomous driving scenarios using a combination of DAI and SAC.
Dataset: I will utilize the ’StreetHazards’ and ’Wildlife’ datasets from the CARLA simulator [1], which provides camera and depth data respectively. Along with a labeled anomaly class which can be utilized to test the DAI algorithm for anomaly detection. [2]
Expected Results: The system should be able to identify and respond to anomalies in real-time while ensuring efficient collaboration between multiple agents. I expect the system to not only detect anomalies quickly due to its use of DAI but detect anomalies spread across the simulated environment efficiently and handle new situations robustly due to its use of mSAC for optimal rollout.
Evaluation: The effectiveness of the system will be mea- sured based on its accuracy in detecting anomalies, false positive/negative rates, detection time, and the efficiency of multi-agent collaboration.
# III. LITERATURE REVIEW
Deep Active Inference: A recent approach for anomaly detection, DAI [3] optimizes the data selection process to focus on the most informative portions, making it efficient for real-time applications. Deep Active Inference has also had comparable results to other algorithms while having a faster stopping time and lower compute cost making it an overall more effective method for anomaly detection compared to the Actor Critic architectures tested. The correlates with neuroscience also is of interest to see how neurally inspired
Soft Actor-Critic: SAC [4], a model-free algorithm, has demonstrated success in multi-agent settings, mSAC or multi-agent Soft Actor-Critic [5] has shown ability to handle large action spaces, while efficiently handling challenges of collaboration and decision-making. SAC with its soft value functions and utilization of the entropy term in its reward function would offer robustness and ability to handle new challenges and disturbances in a dynamic environment surveillance situation.
CARLA Simulator: A popular platform for autonomous driving research, offering realistic urban environments to test and validate various algorithms. Anomaly detection data sets were also utilized in the CARLA simulator. The dataset ’StreetHazards’ and ’Wildlife’ both are used for anomaly detection but are in simulated environments, and take into account temporal context for anomalies which are both useful for the experiments that will be conducted. [1] [2]
Visual Anomaly Detection: Numerous methods have been developed to detect visual anomalies, especially those tailored for dynamic and intricate urban environments. [2]
# IV. TECHNICAL APPROACH
1. Data Preprocessing and Anomaly Detection:
- Data Extraction: Relevant features from the ’StreetHaz- ards’ and ’Wildlife’ datasets are extracted to capture significant indicators of anomalies.
- Data Augmentation: The dataset is augmented to ensure a diverse set of training scenarios, enhancing the model’s robustness.
- Normalization: Features are normalized to ensure a con- sistent scale and to improve convergence during the training phase.
- DAI Training and Validation: The DAI model is trained and validated using the preprocessed data, focusing on the early detection of anomalies.
2. Simulation Setup in CARLA:
- Environment Configuration: The CARLA simulation envi- ronment is configured to replicate real-world scenarios where anomalies might be present, wildlife or pedestrians walking
in certain areas, street hazards, dynamic weather conditions, varying traffic scenarios, and different times of the day.
- Agent Deployment: Autonomous agents are deployed within the simulation environment, equipped with the capa- bility to detect anomalies using the trained DAI model.
3. Integration of DAI with Multi-agent SAC:
- Decomposed SAC: The SAC algorithm is decomposed to facilitate multi-agent collaboration.
- Reward Mechanism Design: A reward system is con- structed where agents are rewarded for detecting anomalies and coordinating responses effectively. The reward mechanism integrates the outputs from the DAI model, emphasizing quick and accurate response to detected anomalies [5].
4. Experiments in CARLA:
- Hide and Seek: Agents are tasked with seeking anomalies in a dynamic environment. The anomalies (represented by ob- jects or scenarios from the datasets) are strategically placed in the environment, challenging agents to detect them efficiently.
- Dynamic Anomaly Detection: The environment is modified in real-time, introducing anomalies dynamically. Agents must adapt and respond to these changes, showcasing the robustness of the integrated system.
- Collaborative Detection: Multiple agents work in tandem to detect anomalies. The experiment evaluates how well agents can collaborate, avoid redundancy, and optimize their paths for efficient anomaly detection.
# V. INTERMEDIATE/PRELIMINARY RESULTS
At this milestone, no results have been achieved as of yet, however preliminary experiments are being conducted on the ’StreetHazards’ dataset to test the DAI algorithm at anomaly detection. This is particularly useful as the dataset was created in the CARLA simulation environment and anomaly detection is a specific use case for this dataset. Once that is complete, the CARLA simulation environment will be set up and the integration with the multi-agent SAC algorithm will be imple- mented and experiments in there will be conducted.
REFERENCES
[1] CARLA Simulator. CARLA: An Open Urban Driving Simulator. Avail- able at: http://carla.org/
[2] Bogdoll, D., Uhlemeyer, S., Kowol, K., & Zo ̈llner, J. M. Perception Datasets for Anomaly Detection in Autonomous Driving: A Survey. 2023 IEEE Intelligent Vehicles Symposium (IV), pages 1–8, 2023. https://doi.org/10.1109/IV55152.2023.10186609
[3] G. Joseph, C. Zhong, M. C. Gursoy, S. Velipasalar, and P. K. Varsh- ney, ”Anomaly Detection via Controlled Sensing and Deep Active In- ference,” arXiv preprint arXiv:2105.06288, 2021. [Online]. Available: https://doi.org/10.48550/arXiv.2105.06288
[4] T. Haarnoja, A. Zhou, K. Hartikainen, G. Tucker, S. Ha, J. Tan, V. Kumar, H. Zhu, A. Gupta, P. Abbeel, and S. Levine, ”Soft Actor-Critic Algorithms and Applications,” arXiv preprint arXiv:1812.05905, 2019. [Online]. Available: https://arxiv.org/abs/1812.05905
[5] Pu, Y., Wang, S., Yang, R., Yao, X., & Li, B. Decomposed Soft Actor-Critic Method for Cooperative Multi-Agent Rein- forcement Learning. arXiv preprint arXiv:2104.06655, 2021. https://doi.org/10.48550/arXiv.2104.06655
