import QtQml 2.12
import QtQuick 2.12
import QtQuick.Controls 2.5
import QtQuick.Layouts 1.12
import QtQuick.Dialogs 1.3

GroupBox {
    property alias loadImageDataPopup: loadImageDataPopup
    property alias loadClassDataPopup: loadClassDataPopup
    property alias loadWeightsDataPopup: loadWeightsDataPopup
    property alias loadButton: loadButton

    title: 'Weights and Checkpoints'
    Layout.fillWidth: true
    GridLayout {
        anchors.fill: parent
        columns: 2
        FileDialog {
            id: weightsFileDialog
            nameFilters: [ "Text files (*.weights)" ]
            title: "Please choose a file"
            folder: shortcuts.home
            property var weightsSet: false
            onAccepted: {
                var weightsFilePath = weightsFileDialog.fileUrls.toString();
                weightsFilePath = weightsFilePath.replace(/^(file:\/{2})/,"");
                yoloBridge.weights_path = weightsFilePath;
                weightsSet = true;
            }
            onRejected: {
            }
        }
        FileDialog {
            id: checkpointFileDialog
            selectFolder: true 
            title: "Please choose a file"
            folder: shortcuts.home
            property var checkpointSet: false
            onAccepted: {
                var checkpointFilePath = checkpointFileDialog.fileUrls.toString();
                checkpointFilePath = checkpointFilePath.replace(/^(file:\/{2})/,"");
                yoloBridge.checkpoint_dir = checkpointFilePath;
                checkpointSet = true;
            }
            onRejected: {
            }
        }
        Button {
            text: "Darknet Weights"
            onClicked: weightsFileDialog.open()
            ToolTip.delay: 1000
            ToolTip.timeout: 5000
            ToolTip.visible: hovered
            ToolTip.text: qsTr(
                "Path to original YOLOv4 Darknet weights \nSelected file: " 
                + yoloBridge.weights_path
                )
            
        }
        Button {
            text: "Checkpoint"
            onClicked: checkpointFileDialog.open()
            ToolTip.delay: 1000
            ToolTip.timeout: 5000
            ToolTip.visible: hovered
            ToolTip.text: qsTr(
                "Path to TF checkpoint directory \nSelected directory: " 
                + yoloBridge.checkpoint_dir)
        }
        CheckBox {
            id: loadWeightsCheckBox
            text: 'Load Darknet Weights'
            checked: false
            onCheckedChanged: yoloBridge.load_darknet_weights = checked
            Layout.alignment: Qt.AlignBottom 
        }
        CheckBox {
            id:  loadCheckpointCheckBox
            text: 'Load Checkpoint'
            checked: false
            onCheckedChanged: yoloBridge.load_checkpoint = checked
            Layout.alignment: Qt.AlignBottom 
        }
        Popup {
            id: weightsPathPopup
            padding: 10
            contentItem: Text {
                text: "please select a weights file or checkpoint before loading"
                color: "#ffffff"
            }
        }
        Popup {
            id: loadImageDataPopup
            padding: 10
            contentItem: Text {
                text: "please select image before running"
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
            id: loadWeightsDataPopup
            padding: 10
            contentItem: Text {
                text: "please load weights or checkpoint before running"
                color: "#ffffff"
            }
        }
        Button {
            id: loadButton
            text: "Load"
            property var weightsLoaded: false
            ToolTip.delay: 1000
            ToolTip.timeout: 5000
            ToolTip.visible: hovered
            ToolTip.text: qsTr("Load weights or checkpoint")
            onClicked: {
                loadImageDataPopup.close()
                loadClassDataPopup.close()
                weightsPathPopup.close()
                if (imagesDetectBox.imageFileDialog.imageDataSet == false){
                    loadImageDataPopup.open()
                }else if (imagesDetectBox.classesFileDialog.classDataSet == false){
                    loadClassDataPopup.open()
                }else if (
                    weightsFileDialog.weightsSet == false &&
                    checkpointFileDialog.checkpointSet == false
                    ){
                        weightsPathPopup.open()
                }else{
                    weightsLoaded = true
                    yoloBridge.yolo_initialize()
                    yoloBridge.yolo_load()
                    }
                }
        }
    }
}