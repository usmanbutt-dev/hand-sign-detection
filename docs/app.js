const metricNodes = document.querySelectorAll("[data-metric]");

fetch("assets/classifier_metrics.json")
  .then((response) => {
    if (!response.ok) throw new Error(`Metrics request failed: ${response.status}`);
    return response.json();
  })
  .then((metrics) => {
    metricNodes.forEach((node) => {
      const key = node.dataset.metric;
      if (!(key in metrics)) return;
      if (key.includes("accuracy")) node.textContent = `${(metrics[key] * 100).toFixed(1)}%`;
      else if (typeof metrics[key] === "number") node.textContent = metrics[key].toFixed(4).replace(/\.0+$/, "");
    });
  })
  .catch(() => {
    // The checked-in fallback values remain visible when opened directly from disk.
  });
