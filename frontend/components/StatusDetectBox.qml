import QtQml 2.12
import QtQuick 2.12
import QtQuick.Controls 2.5
import QtQuick.Layouts 1.12
import QtQuick.Dialogs 1.3

GroupBox {
    title: 'Status'
    Layout.fillWidth: true
    GridLayout {
        anchors.fill: parent
        columns: 2          
        Button {
            id: runDetectionButton
            text: "Run Detection"
            property var imageIterator: 0
            ToolTip.delay: 1000
            ToolTip.timeout: 5000
            ToolTip.visible: hovered
            ToolTip.text: qsTr("Run detection")
            onClicked: {
                weightsDetectBox.loadImageDataPopup.close()
                weightsDetectBox.loadClassDataPopup.close()
                weightsDetectBox.loadWeightsDataPopup.close()
                if (imagesDetectBox.imageFileDialog.imageDataSet == false){
                    weightsDetectBox.loadImageDataPopup.open()
                }else if (imagesDetectBox.classesFileDialog.classDataSet == false){
                    weightsDetectBox.loadClassDataPopup.open()
                }else if (weightsDetectBox.loadButton.weightsLoaded == false){
                    weightsDetectBox.loadWeightsDataPopup.open()
                }else if (imageIterator < yoloBridge.detect_image_path.length){
                    yoloBridge.yolo_detect(imageIterator)
                    image.source = yoloBridge.annotated_image_path[imageIterator];
                    image.visible = true;
                    imageIterator++;
                }else{imageIterator = 0;}
            }
        }
        Text {
            id: imageNumberLabel
            text: "Image: " 
            + runDetectionButton.imageIterator.toString()
            + "/"+yoloBridge.detect_image_path.length.toString()
            Layout.preferredWidth: 150
            wrapMode: Label.WordWrap
            Layout.alignment: Qt.AlignHCenter
            color: "#ffffff"
        }    
    }
}