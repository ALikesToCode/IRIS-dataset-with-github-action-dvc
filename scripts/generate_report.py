import os
import json
import xml.etree.ElementTree as ET
from datetime import datetime
import pandas as pd

def generate_report():
    """Generates a comprehensive CML report from test, coverage, and metrics artifacts."""
    report_parts = []

    # --- Header ---
    report_parts.append('# 🚀 ML Pipeline Report')
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

    # Model Training
    report_parts.append('<details><summary>📈 Model Training & Validation</summary>')
    try:
        with open('artifacts/metrics.json') as f:
            data = json.load(f)
        table = ['\n#### Performance Metrics', '| Metric | Value |', '|:---|:---|']
        for k, v in data.items():
            table.append(f'| {k.replace("_", " ").title()} | {v:.4f} |')
        report_parts.append('\n'.join(table) + '\n')
    except FileNotFoundError:
        report_parts.append('\n❌ **Model training failed or metrics file not found.**\n')
    report_parts.append('</details>')

    # Data Quality
    report_parts.append('<details><summary>📋 Data Quality Report</summary>')
    try:
        df = pd.read_csv('data/iris.csv')
        report_parts.append(f'\n✅ **Data file loaded**: `data/iris.csv`')
        report_parts.append(f'- **Shape**: {df.shape[0]} rows, {df.shape[1]} columns')
        report_parts.append('\n#### Data Preview (first 5 rows)\n```csv\n' + df.head().to_csv(index=False) + '\n```\n')
    except FileNotFoundError:
        report_parts.append('\n❌ **Data file missing**: `data/iris.csv`\n')
    report_parts.append('</details>')

    # --- Footer ---
    report_parts.append('---')
    report_parts.append(f'_Report generated on {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}_')
    
    with open('report.md', 'w') as f:
        f.write('\n\n'.join(report_parts))

if __name__ == '__main__':
    generate_report() 