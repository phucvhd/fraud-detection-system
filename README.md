## Fraud Detection Model

### Resources
- **Dataset**: [Credit Card Fraud Detection – Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

### Architecture
![Service Architecture](assets/service-architecture.png)

### Components

1. **EDA**
   - Data shape, memory usage
   - Basic statistics, total missing value, distribution
   - Transaction amount analysis
   - Time-based pattern analysis
   - Feature correlation
   - Fraud pattern discovery

2. **ML Pipeline**
   - Link: https://github.com/phucvhd/fraud-detection-ml-pipeline

   1. **Preprocessing**
      - Load
      - Validate integrity, consistency, missing value, imbalance
      - Split
      - Scaling

   2. **Imbalance Handling**
      - SMOTE

   3. **Model Training**
      - Decision Tree
      - Random Forest
      - XGBoost

   4. **Model Evaluation**
      - Precision
      - Recall
      - F1 Score
      - PR AUC

   5. **Hyperparameter Tuning**
      - Grid Search
      - Random Search

3. **Transaction generator**
4. **Fraud Detection service**
5. **Kafka architecture**