import QtQuick 2.12
import QtQuick.Controls 2.5
import QtCharts 2.3

ChartView {
    id: chart
    property var giouLossTrainRate: LineSeries {
        axisX: xAxis
        axisY: yAxis
        useOpenGL: true
    }
    property var giouLossTestRate: LineSeries {
        axisX: xAxis
        axisY: yAxis
        useOpenGL: true
    }
    theme: ChartView.ChartThemeDark
    antialiasing: true
    
    ValueAxis{
        id: xAxis
        titleText: 'Iterations'
        min: 1.0
        max: 2.0
    }
    ValueAxis{
        id: yAxis
        titleText: 'GIoU Loss'
        min: 0.0
        max: 1.0
    }

    function updateAxes(point) {
        xAxis.max = Math.max(xAxis.max, point.x)
        xAxis.min = Math.min(xAxis.min, point.x)
        yAxis.max = Math.max(yAxis.max, point.y*1.1)
    }

    function reset() {
        removeAllSeries()
        giouLossTrainRate = createSeries(
            ChartView.SeriesTypeLine, 'Training', xAxis, yAxis
        )
        giouLossTestRate = createSeries(
            ChartView.SeriesTypeLine, 'Test', xAxis, yAxis
        )
        giouLossTrainRate.pointAdded.connect((index) => {
            updateAxes(giouLossTrainRate.at(index))
        })
        giouLossTestRate.pointAdded.connect((index) => {
            updateAxes(giouLossTestRate.at(index))
        })
        xAxis.max = 10
        xAxis.min = 0
        yAxis.max = 1
        yAxis.min = 0
    }

    Component.onCompleted: reset()
}