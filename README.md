# Direct Marketing Optimization

## Author

Ng Wei Xuan

## Deliverables

- doc/presentation.pptx: Presentation or documentation

- training/results/targeted_customers.csv: A Targeted client list specifying selected clients for each offer.

- training/ : all training scripts and unit tests

- training/notebooks: all analysis done for the slides

- inference/ : all inference scripts and unit tests

Production code are in training and inference folder respectively, including dockerfiles and unit tests. Training uses weights and biases as experimental tracking platform. 

## Objective

- Maximize revenue from direct marketing campaigns using the provided dummy data.

- This case study simulates a real-world marketing scenario aimed at optimizing resource allocation to maximize revenue.

## Installation and Requirements

Python 3.8+

Run the following command:

`docker-compose up --build`

Packages (install via pip install -r requirements.txt) via dockerfile