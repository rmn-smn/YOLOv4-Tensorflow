import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Layouts 1.12

import 'components'
import 'components/dashboards'

ApplicationWindow {
    // main application window
    id: window
    visible: true
    title: 'Yolov4 Object Detector'

    Pane {
        id: body
        anchors.fill: parent
        NoteBook {
            Train {}
            Detect {}

        }
    }

    Component.onCompleted: () => {
        width = minimumWidth = body.implicitWidth
        height = minimumHeight = body.implicitHeight
    }
}
