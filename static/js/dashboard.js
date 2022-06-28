(function($) {
    'use strict';
    $(function() {
        if ($('#circleProgress1').length) {
            var bar = new ProgressBar.Circle(circleProgress1, {
                color: '#f44336',
                strokeWidth: 18,
                trailWidth: 18,
                easing: 'easeInOut',
                duration: 1400,
                width: 32,
            });
            bar.animate(.68);
        }
        if ($('#circleProgress2').length) {
            var bar = new ProgressBar.Circle(circleProgress2, {
                color: '#ff9a00',
                strokeWidth: 18,
                trailWidth: 18,
                easing: 'easeInOut',
                duration: 1400,
                width: 32,

            });
            bar.animate(.86);
        }
        if ($("#flotChart").length) {
          var plot = $.plot('#flotChart', [{
              data: flotSampleData4,
              color: '#423898',
              lines: {
                  fillColor: 'rgba(244,243,255,0.5)',
              }
          }, {
              data: flotSampleData3,
              color: '#95d2ec',
              lines: {
                  fillColor: 'rgba(244,243,255,0.1)',
              }
          }], {
              series: {
                  shadowSize: 0,
                  lines: {
                      show: true,
                      lineWidth: 2.2,
                      fill: true
                  }
              },
              grid: {
                  borderWidth: 0,
                  labelMargin: 8,
                  tickColor: '#f8f8f8'
              },
              yaxis: {
                  show: false,
                  min: 0,
                  max: 100,
                  color: '#fff',
                  ticks: [
                      [0, '$25,000'],
                      [20, '$50,000'],
                      [40, '$75,000'],
                      [60, '$100,000'],
                  ],
                  grid: {
                    tickLength: 0,
                  }
              },
              xaxis: {
                  show: true,
                  color: '#fff',
                  ticks: [
                      [20, 'JAN'],
                      [40, 'FEB'],
                      [60, 'MAR'],
                      [80, 'APR'],
                      [100, 'MAY'],
                      [120, 'JUN'],
                      [140, 'JUL'],
                      [160, 'AUG']
                  ],
              }
          });
        }
        if ($("#flotChart-dark").length) {
          var plot = $.plot('#flotChart-dark', [{
              data: flotSampleData4,
              color: '#423898',
              lines: {
                  fillColor: 'rgba(28, 30, 47,0.5)',
              }
          }, {
              data: flotSampleData3,
              color: '#95d2ec',
              lines: {
                  fillColor: 'rgba(28, 30, 47,0.1)',
              }
          }], {
              series: {
                  shadowSize: 0,
                  lines: {
                      show: true,
                      lineWidth: 2.2,
                      fill: true
                  }
              },
              grid: {
                  borderWidth: 0,
                  labelMargin: 8,
                  tickColor: '#1c1e2f'
              },
              yaxis: {
                  show: false,
                  min: 0,
                  max: 100,
                  color: '#222437',
                  ticks: [
                      [0, '$25,000'],
                      [20, '$50,000'],
                      [40, '$75,000'],
                      [60, '$100,000'],
                  ],
                  tickColor: '#eee'
              },
              xaxis: {
                  show: true,
                  color: '#222437',
                  ticks: [
                      [20, 'JAN'],
                      [40, 'FEB'],
                      [60, 'MAR'],
                      [80, 'APR'],
                      [100, 'MAY'],
                      [120, 'JUN'],
                      [140, 'JUL'],
                      [160, 'AUG']
                  ],
              }
          });
        }



        if ($("#marketingTrend").length) {
            var graphGradient = document.getElementById("marketingTrend").getContext('2d');;
            var marketingTrendBg = graphGradient.createLinearGradient(25, 0, 25, 120);
            marketingTrendBg.addColorStop(0, 'rgba(	206, 238, 242, 1)');
            marketingTrendBg.addColorStop(1, 'rgba(255, 255, 255, 1)');

            var marketingTrendBg1 = graphGradient.createLinearGradient(25, 0, 25, 120);
            marketingTrendBg1.addColorStop(0, 'rgba(115, 151, 247, .02)');
            marketingTrendBg1.addColorStop(1, 'rgba(255, 255, 255, 1)');
            var marketingTrendData = {
                labels: ["1", "2", "3", "4", "5", "6", "7", "8", "9"],
                datasets: [{
                        label: 'Critical',
                        data: [13, 12, 11, 10, 10, 11, 12, 13, 13],
                        borderColor: [
                            '#2e5cda'
                        ],
                        backgroundColor: marketingTrendBg1,
                        borderWidth: 2,
                        fill: true,
                    },
                    {
                        label: 'Warning',
                        
                        data: [10, 11, 12, 13, 13, 12, 11, 10, 10],
                        borderColor: [
                          'rgba(53,219,147,.4)',
                        ],
                        borderWidth: 2,
                        fill: true,
                        backgroundColor: marketingTrendBg,
                    }
                ],
            };
            var marketingTrendOptions = {
                scales: {
                    yAxes: [{
                        display: true,
                        gridLines: {
                            drawBorder: false,
                            display: false,
                            drawTicks: false
                        },
                        ticks: {
                            display: false,
                            beginAtZero: true,
                            stepSize: 5
                        }
                    }],
                    xAxes: [{
                        display: false,
                        position: 'bottom',
                        gridLines: {
                            drawBorder: false,
                            display: true,
                            zeroLineColor: '#000',
                            drawTicks: false
                        },
                        ticks: {
                            display: true,
                            beginAtZero: false,
                            stepSize: 5
                        }
                    }],

                },
                maintainAspectRatio: false,
                legend: {
                    display: false,
                    labels: {
                        boxWidth: 0,
                    }
                },
                elements: {
                    point: {
                        radius: 0
                    },
                    line: {
                        tension: .4,
                    },
                },
                tooltips: {
                    backgroundColor: 'rgba(2, 171, 254, 1)',
                }
            };
            var lineChartCanvas = $("#marketingTrend").get(0).getContext("2d");
            var saleschart = new Chart(lineChartCanvas, {
                type: 'line',
                data: marketingTrendData,
                options: marketingTrendOptions
            });
            // var ctx = document.getElementById("marketingTrend");
            // ctx.height = 150;
        }
        if ($("#marketingTrend-dark").length) {
          var graphGradient = document.getElementById("marketingTrend-dark").getContext('2d');;
          var marketingTrendBg = graphGradient.createLinearGradient(25, 0, 25, 110);
          marketingTrendBg.addColorStop(0, 'rgba(33, 191, 6, .2)');
          marketingTrendBg.addColorStop(1, 'rgba(	28, 30, 47, .4)');

          var marketingTrendBg1 = graphGradient.createLinearGradient(25, 0, 25, 110);
          marketingTrendBg1.addColorStop(0, 'rgba(59, 134, 209, .2)');
          marketingTrendBg1.addColorStop(1, 'rgba(	28, 30, 47, .3)');
          var marketingTrendDarkData = {
              labels: ["1", "2", "3", "4", "5", "6", "7", "8", "9"],
              datasets: [{
                      label: 'Critical',
                      data: [13, 12, 11, 10, 10, 11, 12, 13, 13],
                      borderColor: [
                          '#2e5cda'
                      ],
                      backgroundColor: marketingTrendBg1,
                      borderWidth: 1,
                      fill: true,
                  },
                  {
                      label: 'Warning',
                      
                      data: [10, 11, 12, 13, 13, 12, 11, 10, 10],
                      borderColor: [
                          '#7397f7',
                      ],
                      borderWidth: 1,
                      fill: true,
                      backgroundColor: marketingTrendBg,
                  }
              ],
          };
          var marketingTrendDarkOptions = {
              scales: {
                  yAxes: [{
                      display: true,
                      gridLines: {
                          drawBorder: false,
                          display: false,
                          drawTicks: false
                      },
                      ticks: {
                          display: false,
                          beginAtZero: true,
                          stepSize: 5
                      }
                  }],
                  xAxes: [{
                      display: false,
                      position: 'bottom',
                      gridLines: {
                          drawBorder: false,
                          display: true,
                          zeroLineColor: '#000'
                      },
                      ticks: {
                          display: true,
                          beginAtZero: false,
                          stepSize: 5
                      }
                  }],

              },
              maintainAspectRatio: false,
              legend: {
                  display: false,
                  labels: {
                      boxWidth: 0,
                  }
              },
              elements: {
                  point: {
                      radius: 0
                  },
                  line: {
                      tension: .4,
                  },
              },
              tooltips: {
                  backgroundColor: 'rgba(2, 171, 254, 1)',
              }
          };
          var lineChartCanvas = $("#marketingTrend-dark").get(0).getContext("2d");
          var saleschart = new Chart(lineChartCanvas, {
              type: 'line',
              data: marketingTrendDarkData,
              options: marketingTrendDarkOptions
          });
          // var ctx = document.getElementById("marketingTrend");
          // ctx.height = 150;
      }
        if ($("#traffic-platform").length) {
          var trafficPlatformData = {
              labels: ["jan", "feb", "mar", "apr", "may", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "jan", "feb", "mar", "apr", "may", "Jun",],
              datasets: [{
                label: 'Safari',
                data: [55,41,61,83,55,71,79,28,41,55, 37,33, 55,79,55,29,55,35,],
                backgroundColor: [
                  '#ff5d5b', '#ff5d5b', '#ff5d5b', '#ff5d5b', '#ff5d5b', '#ff5d5b', '#ff5d5b', '#ff5d5b', '#ff5d5b', '#ff5d5b', '#ff5d5b', '#ff5d5b','#ff5d5b', '#ff5d5b', '#ff5d5b', '#ff5d5b', '#ff5d5b', '#ff5d5b',
                ],
                borderColor: [
                  '#ff5d5b', '#ff5d5b', '#ff5d5b', '#ff5d5b', '#ff5d5b', '#ff5d5b', '#ff5d5b', '#ff5d5b', '#ff5d5b', '#ff5d5b', '#ff5d5b', '#ff5d5b','#ff5d5b', '#ff5d5b', '#ff5d5b', '#ff5d5b', '#ff5d5b', '#ff5d5b',
                ],
                borderWidth: 1,
                fill: false
              },
              {
                label: 'Chrome',
                data: [15,11,17,3,15,20,3,8,11,15,10,9,15,3,15,8,15,9],
                backgroundColor: [
                  '#83c9d1', '#83c9d1', '#83c9d1', '#83c9d1', '#83c9d1', '#83c9d1', '#83c9d1', '#83c9d1', '#83c9d1', '#83c9d1', '#83c9d1', '#83c9d1','#83c9d1', '#83c9d1', '#83c9d1', '#83c9d1', '#83c9d1', '#83c9d1',
                ],
                borderColor: [
                  '#83c9d1', '#83c9d1', '#83c9d1', '#83c9d1', '#83c9d1', '#83c9d1', '#83c9d1', '#83c9d1', '#83c9d1', '#83c9d1', '#83c9d1', '#83c9d1','#83c9d1', '#83c9d1', '#83c9d1', '#83c9d1', '#83c9d1', '#83c9d1',
                ],
                borderWidth: 1,
                fill: false
              },
              {
                label: 'firefox',
                data: [27,45,19,11,27,6,15,61,45,27,50,55,27,15,27,67,27,52],
                backgroundColor: [
                  'rgba(160, 217, 224, .31)', 'rgba(160, 217, 224, .31)', 'rgba(160, 217, 224, .31)', 'rgba(160, 217, 224, .31)', 'rgba(160, 217, 224, .31)', 'rgba(160, 217, 224, .31)', 'rgba(160, 217, 224, .31)', 'rgba(160, 217, 224, .31)', 'rgba(160, 217, 224, .31)', 'rgba(160, 217, 224, .31)', 'rgba(160, 217, 224, .31)', 'rgba(160, 217, 224, .31)','rgba(160, 217, 224, .31)', 'rgba(160, 217, 224, .31)', 'rgba(160, 217, 224, .31)', 'rgba(160, 217, 224, .31)', 'rgba(160, 217, 224, .31)', 'rgba(160, 217, 224, .31)',
                ],
                borderColor: [
                  'rgba(160, 217, 224, .31)', 'rgba(160, 217, 224, .31)', 'rgba(160, 217, 224, .31)', 'rgba(160, 217, 224, .31)', 'rgba(160, 217, 224, .31)', 'rgba(160, 217, 224, .31)', 'rgba(160, 217, 224, .31)', 'rgba(160, 217, 224, .31)', 'rgba(160, 217, 224, .31)', 'rgba(160, 217, 224, .31)', 'rgba(160, 217, 224, .31)', 'rgba(160, 217, 224, .31)','rgba(160, 217, 224, .31)', 'rgba(160, 217, 224, .31)', 'rgba(160, 217, 224, .31)', 'rgba(160, 217, 224, .31)', 'rgba(160, 217, 224, .31)', 'rgba(160, 217, 224, .31)',
                ],
                borderWidth: 1,
                fill: false
              }
          ]
            };
            var trafficPlatformOptions = {
              scales: {
                xAxes: [{
                  display: false,
                  stacked: true,
                  barPercentage: 0.2,
                  gridLines: {
                    display: false //this will remove only the label
                  },
                  ticks: {
                      beginAtZero: true,
                  }
                }],
                yAxes: [{
                  stacked: true,
                  display: false,
                }]
              },
              maintainAspectRatio: false,
              legend: {
                display: false,
                position: "bottom"
              },
              elements: {
                point: {
                  radius: 0
                }
              },
          
            };
          var barChartCanvas = $("#traffic-platform").get(0).getContext("2d");
          // This will get the first returned node in the jQuery collection.
          var barChart = new Chart(barChartCanvas, {
            type: 'bar',
            data: trafficPlatformData,
            options: trafficPlatformOptions
          });
          // var ctx = document.getElementById("traffic-platform");
          // ctx.height = 150;
        }

        if ($("#performance-indicator").length) {
          var performanceIndicatorData = {
              labels: ["jan", "feb", "mar","apr" ],
              datasets: [{
                label: "Tasks",
                backgroundColor: "#008783",
                data: [171, 186, 82, 64],
                stack: 1,
              },
              {
                label: "Completed Tasks",
                backgroundColor: [
                  pattern.draw('diagonal', 'rgba(0,135,131,.7)'),
                  pattern.draw('diagonal', 'rgba(0,135,131,.7)'),
                  pattern.draw('diagonal', 'rgba(0,135,131,.7)'),
                  pattern.draw('diagonal', 'rgba(0,135,131,.7)'),
                ],
                data: [83, 73, 146, 73],
                stack: 1
              },
              {
                label: "Complaints",
                backgroundColor: "#4f6ada",
                data: [131, 135, 40, 31],
                stack: 2
              },
              {
                label: "Completed Complaints",
                backgroundColor: [
                  pattern.draw('diagonal', 'rgba(79,106,218,.7)'),
                  pattern.draw('diagonal', 'rgba(79,106,218,.7)'),
                  pattern.draw('diagonal', 'rgba(79,106,218,.7)'),
                  pattern.draw('diagonal', 'rgba(79,106,218,.7)'),
                ],
                data: [123, 123, 158, 80],
                stack: 2
              },
              {
                label: "Refferal",
                backgroundColor: "#fdcc8f",
                data: [131, 178, 94, 74],
                stack: 3
              },
              {
                label: "Refferal Completed",
                backgroundColor: [
                  pattern.draw('diagonal', 'rgba(253,204,143,.7)'),
                  pattern.draw('diagonal', 'rgba(253,204,143,.7)'),
                  pattern.draw('diagonal', 'rgba(253,204,143,.7)'),
                  pattern.draw('diagonal', 'rgba(253,204,143,.7)'),
                ],
                data: [123, 80, 162, 80],
                stack: 3
              }
              
          ]
            };
            var performanceIndicatorOptions = {
              scales: {
                xAxes: [{
                  display: true,
                  stacked: true,
                  barPercentage: .6,
                  gridLines: {
                    display: true, //this will remove only the label
                    color:'#f8f8f8',
                  },
                  ticks: {
                      beginAtZero: true,
                  }
                }],
                yAxes: [{
                  stacked: true,
                  display: false,
                  gridLines: {
                    display: true, //this will remove only the label
                  },
                  
                }]
              },
              legend: {
                display: false,
                position: "bottom",
              },
              legendCallback: function(chart) {
                var text = [];
                text.push('<div class="d-flex align-items-center">');
                for (var i = 0; i < chart.data.datasets.length; i=i+2) {
                  console.log(chart.data.datasets[i]);
                  text.push('<span class="legend-label" style="background-color:' + chart.data.datasets[i].backgroundColor + '"></span><p class="text-dark mr-4 mb-0">' + chart.data.datasets[i].label + '</p>');
                }
                text.push('</div>');
                return text.join("");
              },
              elements: {
                point: {
                  radius: 0
                }
              } 
          
            };
          var barChartCanvas = $("#performance-indicator").get(0).getContext("2d");
          // This will get the first returned node in the jQuery collection.
          var barChart = new Chart(barChartCanvas, {
            type: 'bar',
            data: performanceIndicatorData,
            options: performanceIndicatorOptions
          });
          document.getElementById('chart-legends-performance').innerHTML = barChart.generateLegend();
        }
        if ($("#performance-indicator-dark").length) {
          var performanceIndicatorDarkData = {
              labels: ["jan", "feb", "mar","apr" ],
              datasets: [{
                label: "Tasks",
                backgroundColor: "#008783",
                data: [171, 186, 82, 64],
                stack: 1
              },
              {
                label: "Tasks Completed",
                //backgroundColor: fillPattern,
                backgroundColor: [
                  pattern.draw('diagonal', 'rgba(0,135,131,.7)'),
                  pattern.draw('diagonal', 'rgba(0,135,131,.7)'),
                  pattern.draw('diagonal', 'rgba(0,135,131,.7)'),
                  pattern.draw('diagonal', 'rgba(0,135,131,.7)'),
                ],
                data: [83, 73, 146, 73],
                stack: 1
              },
              {
                label: "Complaints",
                backgroundColor: "#4f6ada",
                data: [131, 135, 40, 31],
                stack: 2
              },
              {
                label: "Complaints Completed",
                backgroundColor: [
                  pattern.draw('diagonal', 'rgba(79,106,218,.7)'),
                  pattern.draw('diagonal', 'rgba(79,106,218,.7)'),
                  pattern.draw('diagonal', 'rgba(79,106,218,.7)'),
                  pattern.draw('diagonal', 'rgba(79,106,218,.7)'),
                ],
                data: [123, 123, 158, 80],
                stack: 2
              },
              {
                label: "Refferal",
                backgroundColor: "#fdcc8f",
                data: [131, 178, 94, 74],
                stack: 3
              },
              {
                label: "Refferal Completed",
                backgroundColor: [
                  pattern.draw('diagonal', 'rgba(253,204,143,.7)'),
                  pattern.draw('diagonal', 'rgba(253,204,143,.7)'),
                  pattern.draw('diagonal', 'rgba(253,204,143,.7)'),
                  pattern.draw('diagonal', 'rgba(253,204,143,.7)'),
                ],
                data: [123, 80, 162, 80],
                stack: 3
              }
              
          ]
            };
            var performanceIndicatorDarkOptions = {
              scales: {
                xAxes: [{
                  display: true,
                  stacked: true,
                  barPercentage: .6,
                  gridLines: {
                    display: true, //this will remove only the label
                    color:'#1c1e2f',
                  },
                  ticks: {
                      beginAtZero: true,
                  }
                }],
                yAxes: [{
                  stacked: true,
                  display: false,
                  gridLines: {
                    display: true, //this will remove only the label
                  },
                  
                }]
              },
              legend: {
                display: false,
                position: "bottom",
              },
              legendCallback: function(chart) {
                var text = [];
                text.push('<div class="d-flex align-items-center">');
                for (var i = 0; i < chart.data.datasets.length; i=i+2) {
                  console.log(chart.data.datasets[i]);
                  text.push('<span class="legend-label" style="background-color:' + chart.data.datasets[i].backgroundColor + '"></span><p class="text-light mr-4 mb-0">' + chart.data.datasets[i].label + '</p>');
                }
                text.push('</div>');
                return text.join("");
              },
              elements: {
                point: {
                  radius: 0
                }
              }
          
            };
          var barChartCanvas = $("#performance-indicator-dark").get(0).getContext("2d");
          // This will get the first returned node in the jQuery collection.
          var barChart = new Chart(barChartCanvas, {
            type: 'bar',
            data: performanceIndicatorDarkData,
            options: performanceIndicatorDarkOptions
          });
          document.getElementById('chart-legends-performance').innerHTML = barChart.generateLegend();
        }
        


    });
})(jQuery);