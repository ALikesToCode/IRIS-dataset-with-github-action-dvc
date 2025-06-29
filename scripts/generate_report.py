import os
import json
import xml.etree.ElementTree as ET
from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import joblib
import sys
import sklearn

def generate_report():
    """Generates a professional, data-rich CML report with visualizations."""
    report_parts = []

    # --- Header ---
    report_parts.append('<h1>🚀 ML Pipeline Report</h1>')
    workflow_url = os.getenv('GH_WORKFLOW_URL', '#')
    report_parts.append(f'A summary of the test, validation, and performance results. [View Workflow Run]({workflow_url})')
    report_parts.append('---')

    # --- Metrics Extraction ---
    tests_passed_str, coverage_str, accuracy_str = 'N/A', 'N/A', 'N/A'
    tests_status, cov_status, acc_status = '⚠️', '⚠️', '⚠️'
    total_tests, failures, errors = 0, 0, 0

    try:
        tree = ET.parse('test-results.xml')
        root = tree.getroot()
        suite = root.find('testsuite') if root.find('testsuite') is not None else root
        total_tests = int(suite.get('tests', 0))
        failures = int(suite.get('failures', 0))
        errors = int(suite.get('errors', 0))
        passed = total_tests - failures - errors
        tests_passed_str = f'{passed} / {total_tests}'
        tests_status = '✅' if failures == 0 and errors == 0 else '❌'
    except (FileNotFoundError, ET.ParseError):
        tests_passed_str = 'XML not found'

    try:
        tree = ET.parse('coverage.xml')
        coverage = float(tree.getroot().get('line-rate')) * 100
        coverage_str = f'{coverage:.1f}%'
        if coverage > 80:
            cov_status = '✅'
        elif coverage > 50:
            cov_status = '⚠️'
        else:
            cov_status = '❌'
    except (FileNotFoundError, ET.ParseError, TypeError):
        coverage_str = 'XML not found'

    try:
        with open('artifacts/metrics.json') as f:
            metrics = json.load(f)
        accuracy = metrics.get('accuracy', 0) * 100
        accuracy_str = f'{accuracy:.2f}%'
        if accuracy >= 90:
            acc_status = '✅'
        elif accuracy >= 80:
            acc_status = '⚠️'
        else:
            acc_status = '❌'
    except (FileNotFoundError, json.JSONDecodeError):
        accuracy_str = 'JSON not found'
        
    # --- Summary Table ---
    report_parts.append('## 📊 Summary')
    report_parts.append(f'''
| Status | Metric        | Value          |
| :----: | :------------ | :------------- |
| {tests_status} | Tests Passed  | {tests_passed_str}   |
| {cov_status} | Code Coverage | {coverage_str}   |
| {acc_status} | Model Accuracy  | {accuracy_str}     |
''')
    report_parts.append('---')
    
    # --- Collapsible Details ---
    # Test Results
    report_parts.append('<details><summary>🔬 Unit Test Details</summary>')
    if tests_status == '✅':
        report_parts.append(f'\n✅ **All {total_tests} tests passed successfully!**\n')
    else:
        report_parts.append(f'\n❌ **{failures} failed, {errors} errors out of {total_tests} tests.** Check the workflow logs for details.\n')
    report_parts.append('</details>')

    # Model Performance Section
    report_parts.append('<details><summary><h2>📈 Model Performance</h2></summary>')
    
    # Generate and save confusion matrix
    try:
        model = joblib.load('artifacts/model.joblib')
        X_test = pd.read_csv('artifacts/X_test.csv')
        y_test = pd.read_csv('artifacts/y_test.csv').iloc[:, 0]
        
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=model.classes_, yticklabels=model.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig('artifacts/confusion_matrix.png')
        
        report_parts.append('\n### Confusion Matrix\n')
        report_parts.append('![Confusion Matrix](artifacts/confusion_matrix.png)\n')
        
    except FileNotFoundError:
        report_parts.append('\nCould not generate confusion matrix: model or data not found.\n')

    # Performance Metrics Table
    try:
        with open('artifacts/metrics.json') as f:
            data = json.load(f)
        table = ['\n### Performance Metrics', '| Metric | Value |', '|:---|:---|']
        for k, v in data.items():
            table.append(f'| {k.replace("_", " ").title()} | {v:.4f} |')
        report_parts.append('\n'.join(table) + '\n')
    except FileNotFoundError:
        report_parts.append('\n❌ **Model training failed or metrics file not found.**\n')
    report_parts.append('</details>')

    # Data Quality Section
    report_parts.append('<details><summary>📋 Data Quality Report</summary>')
    try:
        df = pd.read_csv('data/iris.csv')
        report_parts.append(f'\n✅ **Data file loaded**: `data/iris.csv`')
        report_parts.append(f'- **Shape**: {df.shape[0]} rows, {df.shape[1]} columns')
        report_parts.append('\n#### Data Preview (first 5 rows)\n```csv\n' + df.head().to_csv(index=False) + '\n```\n')
    except FileNotFoundError:
        report_parts.append('\n❌ **Data file missing**: `data/iris.csv`\n')
    report_parts.append('</details>')

    # Environment Details Section
    report_parts.append('<details><summary><h2>🛠️ Environment Details</h2></summary>')
    try:
        versions = {
            "Python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "Pandas": pd.__version__,
            "Scikit-learn": sklearn.__version__,
            "Seaborn": sns.__version__,
            "Matplotlib": matplotlib.__version__
        }
        table = ['\n| Library | Version |', '|:---|:---|']
        for lib, version in versions.items():
            table.append(f'| {lib} | {version} |')
        report_parts.append('\n'.join(table) + '\n')
    except Exception as e:
        report_parts.append(f'\nCould not retrieve environment details: {e}\n')
    report_parts.append('</details>')

    # --- Footer ---
    report_parts.append('---')
    report_parts.append(f'_Report generated on {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}_')
    
    with open('report.md', 'w') as f:
        f.write('\n\n'.join(report_parts))

if __name__ == '__main__':
    # We need to save the test data for the confusion matrix
    from main import Config, IrisModelTrainer
    config = Config(data_path='data/iris.csv')
    trainer = IrisModelTrainer(config)
    data = trainer.load_data()
    _, _, X_test, y_test = trainer.prepare_data(data)
    os.makedirs('artifacts', exist_ok=True)
    X_test.to_csv('artifacts/X_test.csv', index=False)
    y_test.to_csv('artifacts/y_test.csv', index=False)
    
    generate_report() 