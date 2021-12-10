import QtQml 2.12
import QtQuick 2.12
import QtQuick.Controls 2.5
import QtQuick.Layouts 1.12
import QtQuick.Dialogs 1.3

//  Epochs  
GroupBox {
    title: 'Settings'
    Layout.fillWidth: true
    GridLayout {
        anchors.fill: parent
        columns: 2
        Label {
            text: 'Total Training Epoches'
            Layout.alignment: Qt.AlignHCenter
        }
        SpinBox {
            id: totalEpochs
            editable: true
            value: 100
            to: 999999
            onValueChanged: yoloBridge.epochs = value
            Component.onCompleted: yoloBridge.epochs = value
            Layout.fillWidth: true
        }
        Label {
            text: 'Warmup Training Epoches'
            Layout.alignment: Qt.AlignHCenter
        }
        SpinBox {
            id: warmupEpochs
            editable: true
            value: 2
            to: 999999
            onValueChanged: yoloBridge.warmup_epochs = value
            Component.onCompleted: yoloBridge.warmup_epochs = value
            Layout.fillWidth: true
        }
        //  Learning Rate  
        Label {
            text: 'Initial Learning Rate'
            Layout.alignment: Qt.AlignHCenter
        }
        SpinBox {
            id: learningRate
            editable: true
            value: 100
            to: 1e6

            property int scale: 1e6
            property int decimals: 8
            property real realValue: value / learningRate.scale

            validator: DoubleValidator {
                bottom: Math.min(learningRate.from, learningRate.to)
                top:  Math.max(learningRate.from, learningRate.to)
            }

            textFromValue: function(value, locale) {
                return (
                    Number(value / learningRate.scale)
                    .toLocaleString(locale, 'f', learningRate.decimals)
                )
            }

            valueFromText: function(text, locale) {
                return Number.fromLocaleString(locale, text) * 100
            }

            onValueChanged: yoloBridge.initial_lr = realValue
            Component.onCompleted: yoloBridge.initial_lr = realValue
            Layout.fillWidth: true
        }
        //  Final Learning Rate  
        Label {
            text: 'Final Learning Rate'
            Layout.alignment: Qt.AlignHCenter
        }
        SpinBox {
            id: finalLearningRate
            editable: true
            value: 1
            to: 1e6

            property int scale: 1e6
            property int decimals: 8
            property real realValue: value / finalLearningRate.scale

            validator: DoubleValidator {
                bottom: Math.min(
                    finalLearningRate.from, finalLearningRate.to)
                top:  Math.max(
                    finalLearningRate.from, finalLearningRate.to)
            }

            textFromValue: function(value, locale) {
                return Number(
                    value / finalLearningRate.scale).toLocaleString(
                    locale, 'f', finalLearningRate.decimals
                    )
            }

            valueFromText: function(text, locale) {
                return Number.fromLocaleString(locale, text) * 100
            }

            onValueChanged: yoloBridge.learning_rate = realValue
            Component.onCompleted: yoloBridge.learning_rate = realValue
            Layout.fillWidth: true
        }
        //  Dataset  
        Label {
            text: 'Batch Size'
            Layout.alignment: Qt.AlignHCenter
        }  
        SpinBox {
            id: batchSize
            editable: true
            value: 4
            to: 100
            onValueChanged: yoloBridge.batch_size = value
            Component.onCompleted: yoloBridge.batch_size = value
            Layout.fillWidth: true
        }  
        Label {
            text: ''
            Layout.alignment: Qt.AlignHCenter
        }          
        CheckBox {
            id: dataAugmentationCheckBox
            text: 'Data Augmentation'
            checked: true
            onCheckedChanged: yoloBridge.data_augmentation = checked
            Component.onCompleted: yoloBridge.data_augmentation = checked
            Layout.alignment: Qt.AlignHCenter
        }
        //  Save Checkpoints                 
        CheckBox {
            id: saveCheckpointsCheckBox
            text: 'Save Checkpoints'
            checked: true
            onCheckedChanged: yoloBridge.save_checkpoints = checked
            Component.onCompleted: yoloBridge.save_checkpoints = checked
            Layout.alignment: Qt.AlignHCenter
        }
        CheckBox {
            id: saveBestCheckpointCheckBox
            text: 'Save Best Checkpoint Only'
            checked: true
            onCheckedChanged: yoloBridge.save_best_checkpoint_only = checked
            Component.onCompleted: yoloBridge.save_best_checkpoint_only = checked
            Layout.alignment: Qt.AlignHCenter
        }
    }
}