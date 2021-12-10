import QtQml 2.12
import QtQuick 2.12
import QtQuick.Controls 2.5
import QtQuick.Layouts 1.12
import QtQuick.Dialogs 1.3

GroupBox {

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
            onAccepted: {
                var path = weightsFileDialog.fileUrls.toString()
                path = path.replace(/^(file:\/{2})/,"");
                yoloBridge.weights_path = path
            }
            onRejected: {
            }
        }
        FileDialog {
            id: checkpointFileDialog
            selectFolder: true 
            title: "Please choose a file"
            folder: shortcuts.home
            onAccepted: {
                var path = checkpointFileDialog.fileUrls.toString()
                path = path.replace(/^(file:\/{2})/,"");
                yoloBridge.checkpoint_dir = path
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
                + yoloBridge.weights_path)

        }
        Button {
            text: "Checkpoint"
            onClicked: checkpointFileDialog.open()
            ToolTip.delay: 1000
            ToolTip.timeout: 5000
            ToolTip.visible: hovered
            ToolTip.text: qsTr(
                "Path to TF checkpoint directory \nSelected file: " 
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
    }
}
