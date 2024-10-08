import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (QApplication, QPushButton, QWidget, QMainWindow,
                             QFileDialog, QGridLayout, QLabel, QVBoxLayout,
                             QHBoxLayout, QMessageBox, QComboBox, QShortcut)
import pyqtgraph as pg
import imageio.v2 as io

# Preparing the environment for the image:
class Image_Interface(QWidget):
    def __init__(self):
        super().__init__()

        # Initialize a QGridLayout
        self.l = QGridLayout(self)
        self.imv = pg.ImageView()
        self.l.addWidget(self.imv)

# Let's define our widget
class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.set_defaults()
        self.setWindowTitle("Image")
        w = QWidget(self)
        self.mainLayout = QVBoxLayout()
        self.setCentralWidget(w)  # QMainWindow takes ownership of the widget pointer and deletes it at the appropriate time.
        w.setLayout(self.mainLayout)

        # Create a Button for menubar and to open File and Save
        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")

        # Open Widget
        self.actionOpen = QtWidgets.QAction(self)
        self.actionOpen.setObjectName("actionOpen")

        # Save Widget
        self.actionSave = QtWidgets.QAction(self)
        self.actionSave.setObjectName("actionSave")

        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addAction(self.actionSave)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menuFile.setTitle("File")

        # Triggering the open Function
        self.actionOpen.setText("Open")
        self.actionOpen.triggered.connect(self.open)

        # Triggering the Save Function
        self.actionSave.setText("Save   Ctrl+s")
        self.actionSave.triggered.connect(self.file_save)

        # Create a Shortcut to save File
        self.saveFile = QtWidgets.QAction(self)
        self.saveFile = QShortcut(QtGui.QKeySequence("Ctrl+s"), self)
        self.saveFile.activated.connect(self.file_save)

        # Setting the minimum size
        self.setMinimumSize(500, 500)

        # Image Widget
        self.imageViewer = Image_Interface()
        self.mainLayout.addWidget(self.imageViewer)

        # Enabling the drop events
        self.setAcceptDrops(True)

    # Set default values for the application
    def set_defaults(self):
        self.status = self.statusBar()
        self.im = None  # Initialize the variable containing the image

    # Clear items inside a Layout
    def clearItems(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
                else:
                    self.clearItems(item.layout())

    # Clear layouts inside layouts
    def clearLayouts(self, layout):
        self.clearItems(layout)
        for i in reversed(range(layout.count())):
            layout_item = layout.itemAt(i)
            self.clearItems(layout_item.layout())
            layout.removeItem(layout_item)

    # Function to open and load an image
    def open(self):
        fn, _ = QFileDialog.getOpenFileName(filter="*.png *.jpg")

        if fn:
            self.status.showMessage(fn)
            self.im = io.imread(fn)
            self.imageViewer.imv.setImage(self.im)
            QMessageBox.information(self,
                                    "file loaded",
                                    "Image successfully loaded!")
        else:
            QMessageBox.critical(self,
                                "Meaningful error",
                                "Something went wrong!")

    # Function to save an Image using Ctrl+S shortcut
    def file_save(self):
        fn, _ = QFileDialog.getSaveFileName(filter="*.png *.jpg")
        if fn:
            self.status.showMessage(fn)
            self.im = io.imsave(fn, self.im)
            QMessageBox.information(self,
                                    "file saved",
                                    "Image was saved successfully!")
        else:
            QMessageBox.critical(self,
                                "Meaningful error",
                                "Something went wrong!")

    # Function used for to drag_and_drop the image to the window
    def dragEnterEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasImage:
            image_path = [u.toLocalFile() for u in event.mimeData().urls()]
            for f in image_path:
                if f:
                    self.status.showMessage(f)
                    self.im = io.imread(f)
                    self.imageViewer.imv.setImage(self.im)
                    QMessageBox.information(self,
                                            "file saved",
                                            "Image successfully loaded!!")
                else:
                    QMessageBox.critical(self,
                                        "Meaningful error",
                                        "Something went wrong!")
            event.accept()
        else:
            event.ignore()

    # Giving back feedback to the user
    def showdialog(self, flag):
        msg = QMessageBox()
        if flag:
            msg.setIcon(QMessageBox.Critical)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.setDefaultButton(QMessageBox.Retry)
            msg.setWindowTitle("Error")
            msg.setText("Error trying to save the image!")
            msg.setInformativeText("Image could not be saved")
            returnValue = msg.exec()
        else:
            msg.setWindowTitle("Info")
            msg.setText("Image was saved successfully!")
            msg.setIcon(QMessageBox.Information)
            msg.setStandardButtons(QMessageBox.Ok)
            returnValue = msg.exec()


def main():
    app = QApplication(sys.argv)
    win = MyWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
