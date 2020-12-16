const oLineChart = new Chartist.Line(
  '#chart',
  // Data.
  {
    labels: [],
    series: []
  },
  // Options.
  {
    fullWidth: true,
    height: '80vh',
    chartPadding: {
      right: 20,
      left: 20
    },
    low: 0,
    axisX: {
      // Draw only every hundredth label of the x axis.
      labelInterpolationFnc: function(value, index) {
        return index % 100 === 0 ? value : '';
      }
    }
  }
);

const oLineChartData = {
  labels: [],
  series: []
};

function handleMetricsEvent(oEvent) {
  // Get metrics of model training.
  const oEventData = JSON.parse(oEvent.data);
  // Set timestamp values as x axis data.
  oLineChartData.labels = oEventData.payload.map(oItem => {
    const oDate = new Date(oItem.timestamp);
    const sHours = oDate.getHours().toString();
    const sMinutes = oDate.getMinutes() < 10 ? `0${oDate.getMinutes()}` : oDate.getMinutes().toString();
    return `${sHours}:${sMinutes}`;
  });
  // Clear y axis data.
  oLineChartData.series.length = 0;
  // Add 'loss' values.
  const aLoss = oEventData.payload.map(oItem => oItem.logs.loss);
  oLineChartData.series.push(aLoss);
  // Add 'accuracy' values (if provided).
  const aAccuracy = oEventData.payload.map(oItem => oItem.logs.acc);
  if (aAccuracy.length > 0) {
    oLineChartData.series.push(aAccuracy);
  }
  // Redraw chart with updated metrics.
  oLineChart.update(oLineChartData);
}

let oEventSource = new EventSource('epoch/boston');
function updateMetricsEventSource(oEvent) {
  const sEventSourcePath = oEvent.target.value;
  if (oEventSource) {
    oEventSource.close();
  }
  oEventSource = new EventSource(sEventSourcePath);
  oEventSource.onmessage = handleMetricsEvent;
}

