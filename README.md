<h1>Fraud Detection Model</h1>

<h2>Resources</h2>
Datasets: Credit Card Fraud Detection - Kaggle

<h2>Architecture</h2>
![Description](assets/service-architecture.png)

<h2>Components</h2>
1. **EDA**
   - Data shape, memory usage
   - Basic statistics, total missing value, distribution
   - Transaction amount analysis
   - Time-based pattern analysis
   - Feature correlation
   - Fraud pattern discovery
2. **Preprocessing**
   - Load
   - Validate integrity, consistency, missing value, imbalance
   - Split
   - Scaling
3. **Imbalance handling**
   - SMOTE
4. **Model training**
   - Decision Tree
   - Random forest
   - XGBoost
5. **Model evaluation**
   - Precision
   - Recall
   - F1 score
   - pr auc
6. **Hyper tuning**
   - Grid search
   - Random search