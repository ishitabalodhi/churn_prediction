# Customer Churn Prediction in the Banking Sector

This project presents a machine learning-based approach to predict customer churn in the banking industry and implement effective customer retention strategies.
Customer churn refers to customers leaving or discontinuing a service. In the banking sector, retaining existing customers is more cost-effective than acquiring new ones. This project uses machine learning models to predict churn based on customer behavior, demographics, and transaction history.

## Machine Learning Models

Multiple models were implemented and evaluated:
- **Support Vector Machine (SVM)** performed well for the majority class but failed to capture churn cases due to class imbalance.
- **Decision Tree (DT)** improved recall but introduced high false positives.
- **Random Forest (RF)** achieved high accuracy and precision but was biased toward the non-churn class.

## Proposed Model: XGBoost + Random Forest with Stratified 10-Fold Cross-Validation

To overcome class imbalance and improve predictive accuracy, a hybrid ensemble model was developed by combining XGBoost and Random Forest, along with Stratified 10-Fold Cross-Validation. This setup ensures each fold preserves the class distribution, resulting in a more robust evaluation.

Hyperparameter tuning was done using **Optuna** (Bayesian Optimization), optimizing parameters like max depth, learning rate, and subsample ratios. The final model achieved an accuracy of **97.1%**, with improved F1-score and a significant reduction in false positives and false negatives.

## Customer Retention Strategies

Based on predicted churn probabilities, customers are segmented into risk categories:

- **Very High Risk (>80%)**: 30% discount + VIP support
- **High Risk (60–80%)**: 15% discount + premium services
- **Moderate Risk (40–60%)**: Personalized emails and perks
- **Low Risk (<40%)**: Regular engagement

These strategies help banks retain valuable customers and minimize revenue loss through targeted engagement. 
Additionally, **churn probability for individual clients** can be calculated, enabling customer-specific risk analysis and intervention.

## Libraries and Tools Used

- `scikit-learn`
- `xgboost`
- `optuna`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `python3`

## Conclusion

By integrating ensemble learning and risk-based segmentation, this project demonstrates a practical and scalable approach to reducing customer churn in the banking domain.
