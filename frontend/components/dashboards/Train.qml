import QtQml 2.12
import QtQuick 2.12
import QtQuick.Controls 2.5
import QtQuick.Layouts 1.12
import QtQuick.Dialogs 1.3

import '..'

Page {
    // training window page
    name: 'Train'
    ColumnLayout {
        DatasetTrainBox{id: datasetTrainBox}
        WeightsTrainBox{id: weightsTrainBox}
        SettingsTrainBox{id: settingsTrainBox}
        StatusTrainBox{id: statusTrainBox}
        StatisticsTrainBox{id: statisticsTrainBox}
    }
    ColumnLayout {
        GroupBox {
            title: 'Charts'
            Layout.fillWidth: true
            Layout.fillHeight: true
            GridLayout {
                anchors.fill: parent
                columns: 2
                GiouLossChart {
                    width: 400
                    id: giouLossChart
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                }
                ConfLossChart {
                    width: 400
                    id: confLossChart
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                }
                ProbLossChart {
                    width: 400
                    id: probLossChart
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                }
                TotalLossChart {
                    width: 400
                    id: totalLossChart
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                }
            }
        }
    }
}