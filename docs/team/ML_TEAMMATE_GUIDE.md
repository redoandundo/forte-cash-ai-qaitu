# ML teammate handoff

Current ML repair status, measured quality and limits: [training report](../../reports/training_report.md).

Exact install, training, calibration, export and inference commands: [ML to inference](../handoffs/ml-to-inference.md).

The branch owns ML code, configurations, tests and aggregate reports. Model bundles and source annotations are distributed outside ordinary Git. Use one versioned bundle with its own manifest, preprocessing, labels, threshold and checksums. Both evaluated recipes currently fail the quality gate; keep the original v3 available as fallback.
