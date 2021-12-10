import QtQml 2.12
import QtQuick 2.12
import QtQuick.Controls 2.5
import QtQuick.Layouts 1.12
import QtQuick.Dialogs 1.3

GroupBox {
    title: 'Statistics'
    Layout.fillWidth: true
    Layout.fillHeight: true
    GridLayout {
        anchors.left: parent.left
        anchors.right: parent.right
        columns: 2
        Label {
            text: 'Current Training Epoch'
            Layout.alignment: Qt.AlignHCenter
        }
        Label {
            text: currentEpoch()
            horizontalAlignment: Text.AlignHCenter
            Layout.fillWidth: true
            function currentEpoch() {
                const epoch = yoloBridge.current_epoch
                if (isNaN(epoch))
                    return 0
                return (
                    epoch.toString() + '/' 
                    + yoloBridge.epochs.toString())
            }
        }
        Label {
            text: 'Training Iteration'
            Layout.alignment: Qt.AlignHCenter
        }
        Label {
            text: currentIter()
            horizontalAlignment: Text.AlignHCenter
            Layout.fillWidth: true
            function currentIter() {
                const iter = yoloBridge.current_iter%yoloBridge.steps_per_epoch 
                if (isNaN(iter))
                    return 0
                return (
                    iter.toString() + '/' 
                    + yoloBridge.steps_per_epoch.toString()
                    )
            }
        }
        Label {
            text: 'Learning Rate'
            Layout.alignment: Qt.AlignHCenter
        }
        Label {
            text: currentLr()
            horizontalAlignment: Text.AlignHCenter
            Layout.fillWidth: true
            function currentLr() {
                const learning_rate  = yoloBridge.current_learning_rate 
                if (isNaN(learning_rate))
                    return 0
                return learning_rate.toFixed(8)
            }
        }
        Label {
            text: 'Training GIoU Loss'
            Layout.alignment: Qt.AlignHCenter
        }
        Label {
            text: currentGiou()
            horizontalAlignment: Text.AlignHCenter
            Layout.fillWidth: true
            function currentGiou() {
                const giou_loss = yoloBridge.current_train_giou_loss
                if (isNaN(giou_loss))
                    return 0
                return giou_loss.toFixed(2);
            }
            onTextChanged: () => {
                if (yoloBridge.current_iter > 0){
                    giouLossChart.giouLossTrainRate.append(
                        yoloBridge.current_iter,
                        yoloBridge.current_train_giou_loss
                    )
                }
                if (yoloBridge.current_epoch > 0){
                    giouLossChart.giouLossTestRate.append(
                        yoloBridge.current_epoch*yoloBridge.steps_per_epoch,
                        yoloBridge.current_test_giou_loss
                    )
                }
            }
        }
        Label {
            text: 'Training Confidence Loss'
            Layout.alignment: Qt.AlignHCenter
        }
        Label {
            text: currentConf()
            horizontalAlignment: Text.AlignHCenter
            Layout.fillWidth: true
            function currentConf() {
                const conf_loss = yoloBridge.current_train_conf_loss
                if (isNaN(conf_loss))
                    return 0
                return conf_loss.toFixed(2);
            }
            onTextChanged: () => {
                if (yoloBridge.current_iter > 0){
                    confLossChart.confLossTrainRate.append(
                        yoloBridge.current_iter,
                        yoloBridge.current_train_conf_loss
                    )
                }
                if (yoloBridge.current_epoch > 0){
                    confLossChart.confLossTestRate.append(
                        yoloBridge.current_epoch*yoloBridge.steps_per_epoch,
                        yoloBridge.current_test_conf_loss
                    )
                }
            }
        }
        Label {
            text: 'Training Probability Loss'
            Layout.alignment: Qt.AlignHCenter
        }
        Label {
            text: currentProb()
            horizontalAlignment: Text.AlignHCenter
            Layout.fillWidth: true
            function currentProb() {
                const conf_loss = yoloBridge.current_train_prob_loss
                if (isNaN(conf_loss))
                    return 0
                return conf_loss.toFixed(2);
            }
            onTextChanged: () => {
                if (yoloBridge.current_iter > 0){
                    probLossChart.probLossTrainRate.append(
                        yoloBridge.current_iter,
                        yoloBridge.current_train_prob_loss
                    )
                }
                if (yoloBridge.current_epoch > 0){
                    probLossChart.probLossTestRate.append(
                        yoloBridge.current_epoch*yoloBridge.steps_per_epoch,
                        yoloBridge.current_test_prob_loss
                    )
                }
            }
        }
        Label {
            text: 'Training Total Loss'
            Layout.alignment: Qt.AlignHCenter
        }
        Label {
            text: currentTotal()
            horizontalAlignment: Text.AlignHCenter
            Layout.fillWidth: true
            function currentTotal() {
                const total_loss = yoloBridge.current_train_total_loss
                if (isNaN(total_loss))
                    return 0
                return total_loss.toFixed(2);
            }
            onTextChanged: () => {
                if (yoloBridge.current_iter > 0){
                    totalLossChart.totalLossTrainRate.append(
                        yoloBridge.current_iter,
                        yoloBridge.current_train_total_loss
                    )
                }
                if (yoloBridge.current_epoch > 0){
                    totalLossChart.totalLossTestRate.append(
                        yoloBridge.current_epoch*yoloBridge.steps_per_epoch,
                        yoloBridge.current_test_total_loss
                    )
                }
            }
        }
    }
}
