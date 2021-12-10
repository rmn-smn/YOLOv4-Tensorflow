import QtQml 2.12
import QtQuick 2.12
import QtQuick.Controls 2.5
import QtQuick.Layouts 1.12
import QtQuick.Dialogs 1.3

GroupBox {
    property alias trainDataFileDialog: trainDataFileDialog
    property alias testDataFileDialog: testDataFileDialog
    property alias classesFileDialog: classesFileDialog
    
    title: 'Dataset'
    Layout.fillWidth: true
    GridLayout {
        anchors.fill: parent
        columns: 3
        FileDialog {
            id: trainDataFileDialog
            nameFilters: [ "Text files (*.txt)" ]
            title: "Please choose a file"
            folder: shortcuts.home
            property var trainDataSet: false
            onAccepted: {
                var path = trainDataFileDialog.fileUrls.toString()
                path = path.replace(/^(file:\/{2})/,"");
                yoloBridge.train_annotations = path
                trainDataSet = true

            }
            onRejected: {
            }
        }
        FileDialog {
            id: testDataFileDialog
            nameFilters: [ "Text files (*.txt)" ]
            title: "Please choose a file"
            folder: shortcuts.home
            property var testDataSet: false
            onAccepted: {
                var path = testDataFileDialog.fileUrls.toString()
                path = path.replace(/^(file:\/{2})/,"");
                yoloBridge.test_annotations = path
                testDataSet = true
            }
            onRejected: {
            }
        }
        FileDialog {
            id: classesFileDialog
            nameFilters: [ "Text files (*.names)" ]
            title: "Please choose a file"
            folder: shortcuts.home
            property var classDataSet: false
            onAccepted: {
                var path = classesFileDialog.fileUrls.toString()
                path = path.replace(/^(file:\/{2})/,"");
                yoloBridge.classes = path
                classDataSet = true
            }
            onRejected: {
            }
        }
        Button {
            text: "Train Set"
            onClicked: trainDataFileDialog.open()
            ToolTip.delay: 1000
            ToolTip.timeout: 5000
            ToolTip.visible: hovered
            ToolTip.text: qsTr(
                "Path to train set .txt \nSelected file: " + yoloBridge.train_annotations)
        }
        Button {
            text: "Test Set"
            onClicked: testDataFileDialog.open()
            ToolTip.delay: 1000
            ToolTip.timeout: 5000
            ToolTip.visible: hovered
            ToolTip.text: qsTr(
                "Path to test set .txt \nSelected file: " + yoloBridge.test_annotations)
        }
        Button {
            text: "Classes"
            onClicked: classesFileDialog.open()
            ToolTip.delay: 1000
            ToolTip.timeout: 5000
            ToolTip.visible: hovered
            ToolTip.text: qsTr("Path to classes file \nSelected file: " + yoloBridge.classes)
        }
    }
}