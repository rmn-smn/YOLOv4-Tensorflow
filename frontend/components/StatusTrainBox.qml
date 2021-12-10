import QtQml 2.12
import QtQuick 2.12
import QtQuick.Controls 2.5
import QtQuick.Layouts 1.12
import QtQuick.Dialogs 1.3

GroupBox {
    title: 'Status'
    Layout.fillWidth: true
    Layout.fillHeight: true
    GridLayout {
        anchors.left: parent.left
        anchors.right: parent.right
        columns: 3
        Popup {
            id: weightsPathPopup
            padding: 10
            contentItem: Text {
                text: "please select a weights file or checkpoint before loading"
                color: "#ffffff"
            }
        }
        Popup {
            id: loadTrainTestDataPopup
            padding: 10
            contentItem: Text {
                text: "please select train and test data before running"
                color: "#ffffff"
            }
        }
        Popup {
            id: loadClassDataPopup
            padding: 10
            contentItem: Text {
                text: "please select classes file before running"
                color: "#ffffff"
            }
        }
        Popup {
            id: loadWeightsPopup
            padding: 10
            contentItem: Text {
                text: "please load weights or checkpoint before running"
                color: "#ffffff"
            }
        }
        Button {
            id: startButton
            text: "Run Training"
            onClicked: () => {
                loadTrainTestDataPopup.close()
                loadClassDataPopup.close()
                loadWeightsPopup.close()
                if (
                    datasetTrainBox.trainDataFileDialog.trainDataSet == false || 
                    datasetTrainBox.testDataFileDialog.testDataSet == false){
                        loadTrainTestDataPopup.open()
                }else if (datasetTrainBox.classesFileDialog.classDataSet == false){
                    loadClassDataPopup.open()
                }else{
                    giouLossChart.reset()
                    probLossChart.reset()
                    confLossChart.reset()
                    totalLossChart.reset()
                    yoloBridge.start_training()
                }
            }
        }
        Button {
            id: stopButton
            text: "Stop Training"
                onClicked: () => {
                    yoloBridge.stop_training()
                }
        }
        Text {
            id: statusLabel
            text: status()
            Layout.preferredWidth: 150
            wrapMode: Label.WordWrap
            Layout.alignment: Qt.AlignHCenter
            color: "#5af27b"
            function status() {
                statusLabel.color = "#5af27b"
                status = yoloBridge.status.toString()
                if (
                    status == 'interrupting training...' || 
                    status == 'stopped training'
                ){
                    statusLabel.color = "#f25a5a"  
                }else if (status == 'detector not yet initialized'){
                    statusLabel.color = "#ffffff"
                }
                return status
            }
        }
    }
}