import QtQml 2.12
import QtQuick 2.12
import QtQuick.Controls 2.5
import QtQuick.Layouts 1.12
import QtQuick.Dialogs 1.3

GroupBox {
    property alias imageFileDialog: imageFileDialog
    property alias classesFileDialog: classesFileDialog
    
    title: 'Image'
    Layout.fillWidth: true
    GridLayout {
        anchors.fill: parent
        columns: 3
        FileDialog {
            id: imageFileDialog
            //nameFilters: [ "Image files (*.jpg *.png, *.tiff, *.tif)" ]
            title: "Please choose a file"
            folder: shortcuts.home
            selectMultiple: true
            property var imageDataSet: false
            onAccepted: {
                var path = imageFileDialog.fileUrls.toString().split(",")
                for(var i=0; i < path.length; i++) { 
                    path[i] = path[i].replace(/^(file:\/{2})/,"");
                    }
                yoloBridge.detect_image_path = path
                imageDataSet = true
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
            text: "Select Image"
            id: imageButton
            onClicked: imageFileDialog.open()
            ToolTip.delay: 1000
            ToolTip.timeout: 5000
            ToolTip.visible: hovered
            ToolTip.text: qsTr(
                "Path to image files \nSelected files: " 
                + yoloBridge.detect_image_path)
        }
        Button {
            text: "Classes"
            id: classButton
            onClicked: classesFileDialog.open()
            ToolTip.delay: 1000
            ToolTip.timeout: 5000
            ToolTip.visible: hovered
            ToolTip.text: qsTr("Path to classes file \nSelected file: " 
                + yoloBridge.classes)
        }
    }
}
