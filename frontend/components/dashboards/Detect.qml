import QtQml 2.12
import QtQuick 2.12
import QtQuick.Controls 2.5
import QtQuick.Layouts 1.12
import QtQuick.Dialogs 1.3

import '..'


Page {
    // detection window page
    name: 'Detect'
    ColumnLayout {
        ImagesDetectBox{id: imagesDetectBox}
        WeightsDetectBox{id: weightsDetectBox}
        StatusDetectBox{id: statusDetectBox}
    }
    ColumnLayout {
         GroupBox {
             title: 'Detection'
             width: 800
             Layout.preferredWidth: 800
             Layout.fillWidth: true
             Layout.fillHeight: true
            Image {
                id: image
                width: 800
                Layout.preferredWidth: 800
                Layout.fillWidth: true
                Layout.fillHeight: true
                fillMode: Image.PreserveAspectFit
                visible: true //hidden by default
            }       
        }        
    }
}