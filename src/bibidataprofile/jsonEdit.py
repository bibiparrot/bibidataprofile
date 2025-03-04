# import ast
import json

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QWidget
from PyQt5 import QtWidgets, QtCore, QtGui


def prettifyJson(jsonText):
    """
    Takes a JSON string and returns a prettified version with proper indentation.

    Args:
        jsonText (str): A valid JSON string

    Returns:
        str: Formatted JSON string with 4-space indentation

    Raises:
        json.JSONDecodeError: If the input string is not valid JSON
    """
    # Parse the JSON string
    parsed_json = json.loads(jsonText)
    # Convert back to a string with nice formatting
    prettified = json.dumps(parsed_json, indent=4, sort_keys=False, ensure_ascii=False)
    return prettified


class QJsonEdit(QWidget):
    def __init__(self, parent=None, jsonDict=None):
        super().__init__(parent)
        self.font = QFont("Microsoft YaHei", 12)
        self.setupUi()

        self.ui_tree_view = QJsonView()
        # self.ui_tree_view.setStyleSheet('QWidget{font: 12 "Microsoft YaHei";}')
        self.ui_tree_view.setFont(self.font)

        self.ui_grid_layout.addWidget(self.ui_tree_view, 1, 0)
        if jsonDict is None:
            jsonDict = {}
        root = QJsonNode.load(jsonDict)
        self._model = QJsonModel(root, self)

        # proxy model
        self._proxyModel = QtCore.QSortFilterProxyModel(self)
        self._proxyModel.setSourceModel(self._model)
        self._proxyModel.setDynamicSortFilter(False)
        self._proxyModel.setSortRole(QJsonModel.sortRole)
        self._proxyModel.setFilterCaseSensitivity(QtCore.Qt.CaseInsensitive)
        self._proxyModel.setFilterRole(QJsonModel.filterRole)
        self._proxyModel.setFilterKeyColumn(0)

        self.ui_tree_view.setModel(self._proxyModel)

        self.ui_filter_edit.textChanged.connect(self._proxyModel.setFilterRegExp)
        self.ui_out_btn.clicked.connect(self.updateBrowser)
        self.ui_update_btn.clicked.connect(self.updateModel)

        # Json Viewer
        self.highlighter = JsonHighlighter(self.ui_text_edit.document())
        self.updateBrowser()

    def getDict(self):
        return self.ui_tree_view.asDict(None)

    def setDict(self, jsonDict):
        root = QJsonNode.load(jsonDict)
        self._model = QJsonModel(root, self)
        self._proxyModel.setSourceModel(self._model)
        self.ui_tree_view.expandAll()
        self.ui_text_edit.setPlainText(json.dumps(jsonDict, indent=4, sort_keys=False, ensure_ascii=False))

    def setJson(self, text):
        jsonDict = json.loads(text)  # Text to Json.
        self.setDict(jsonDict)

    def updateModel(self):
        text = self.ui_text_edit.toPlainText()
        self.setJson(text)

    def updateBrowser(self):
        self.ui_text_edit.clear()
        jsonDict = self.ui_tree_view.asDict(None)
        jsonText = json.dumps(jsonDict, indent=4, sort_keys=False, ensure_ascii=False)
        self.ui_text_edit.setPlainText(str(jsonText))

    def setupUi(self):
        # Main layout
        self.ui_grid_layout = QtWidgets.QGridLayout(self)
        self.ui_filter_edit = QtWidgets.QLineEdit(self)
        self.ui_filter_edit.setFont(self.font)
        self.ui_grid_layout.addWidget(self.ui_filter_edit, 0, 0, 1, 1)
        self.ui_text_edit = QtWidgets.QPlainTextEdit(self)
        self.ui_text_edit.setFont(self.font)
        self.ui_grid_layout.addWidget(self.ui_text_edit, 0, 2, 2, 1)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setContentsMargins(-1, -1, 0, -1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem)
        self.ui_out_btn = QtWidgets.QPushButton(self)
        self.ui_out_btn.setFont(self.font)
        self.verticalLayout_2.addWidget(self.ui_out_btn)
        self.ui_update_btn = QtWidgets.QPushButton(self)
        self.ui_update_btn.setFont(self.font)
        self.verticalLayout_2.addWidget(self.ui_update_btn)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem1)
        self.ui_grid_layout.addLayout(self.verticalLayout_2, 1, 1, 1, 1)
        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.ui_out_btn.setText(_translate("QJsonEdit", ">>"))
        self.ui_update_btn.setText(_translate("QJsonEdit", "<<"))


class QJsonView(QtWidgets.QTreeView):
    dragStartPosition = None

    def __init__(self):
        """
        Initialization
        """
        super(QJsonView, self).__init__()

        self._clipBroad = ''

        # set flags
        self.setSortingEnabled(True)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setUniformRowHeights(True)

        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.openContextMenu)

    def setModel(self, model):
        """
        Extend: set the current model and sort it

        :param model: QSortFilterProxyModel. model
        """
        super(QJsonView, self).setModel(model)
        self.model().sort(0, QtCore.Qt.AscendingOrder)

    def openContextMenu(self):
        """
        Custom: create a right-click context menu
        """
        contextMenu = QtWidgets.QMenu()

        indices = self.getSelectedIndices()
        # no selection
        if not indices:
            addAction = contextMenu.addAction('add entry')
            addAction.triggered.connect(self.customAdd)

            clearAction = contextMenu.addAction('clear')
            clearAction.triggered.connect(self.clear)
        else:
            removeAction = contextMenu.addAction('remove entry(s)')
            removeAction.triggered.connect(lambda: self.remove(indices))

            copyAction = contextMenu.addAction('copy entry(s)')
            copyAction.triggered.connect(self.copy)

        # single selection
        if len(indices) == 1:
            index = indices[0]

            # only allow add when the index is a dictionary or list
            if index.internalPointer().dtype in [list, dict]:
                addAction = contextMenu.addAction('add entry')
                addAction.triggered.connect(lambda: self.customAdd(index=index))

                if self._clipBroad:
                    pasteAction = contextMenu.addAction('paste entry(s)')
                    pasteAction.triggered.connect(lambda: self.paste(index))

        contextMenu.exec_(QtGui.QCursor().pos())

    # helper methods

    def getSelectedIndices(self):
        """
        Custom: get source model indices of the selected item(s)

        :return: list of QModelIndex. selected indices
        """
        indices = self.selectionModel().selectedRows()
        return [self.model().mapToSource(index) for index in indices]

    def asDict(self, indices):
        """
        Custom: serialize specified model indices to dictionary

        :param indices: list of QModelIndex. root indices
        :return: dict. output dictionary
        """
        output = dict()
        if not indices:
            output = self.model().sourceModel().asDict()
        else:
            for index in indices:
                output.update(self.model().sourceModel().asDict(index))
        return output

    # overwrite drag and drop

    def mousePressEvent(self, event):
        """
        Override: record mouse click position
        """
        super(QJsonView, self).mousePressEvent(event)
        if event.button() == QtCore.Qt.LeftButton:
            self.dragStartPosition = event.pos()

    def mouseMoveEvent(self, event):
        """
        Override: instantiate custom drag object when dragging with left-click
        """
        if not event.buttons():
            return

        if not event.buttons() == QtCore.Qt.LeftButton:
            return

        if (event.pos() - self.dragStartPosition).manhattanLength() \
                < QtWidgets.QApplication.startDragDistance():
            return

        if self.selectionModel().selectedRows():
            drag = QtGui.QDrag(self)
            mimeData = QtCore.QMimeData()

            selected = self.asDict(self.getSelectedIndices())
            mimeData.setText(str(selected))
            drag.setMimeData(mimeData)

            drag.exec_()

    def dragEnterEvent(self, event):
        """
        Override: allow dragging only for certain drag object
        """
        data = event.mimeData()
        if data.hasText():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        """
        Override: disable dropping to certain model index based on node type
        """
        data = event.mimeData()
        if data.hasText():
            event.acceptProposedAction()

        dropIndex = self.indexAt(event.pos())
        dropIndex = self.model().mapToSource(dropIndex)

        # not allowing drop to non dictionary or list
        if not dropIndex == QtCore.QModelIndex():
            if dropIndex.internalPointer().dtype not in [list, dict]:
                event.ignore()

    def dropEvent(self, event):
        """
        Override: customize drop behavior to move for internal drag & drop
        """
        dropIndex = self.indexAt(event.pos())
        dropIndex = self.model().mapToSource(dropIndex)

        data = event.mimeData()
        self.remove(self.getSelectedIndices())
        self.add(data.text(), dropIndex)
        event.acceptProposedAction()

    # custom behavior

    def remove(self, indices):
        """
        Custom: remove node(s) of specified indices

        :param indices: QModelIndex. specified indices
        """
        for index in indices:
            currentNode = index.internalPointer()
            position = currentNode.row()

            # let the model know we are removing
            self.model().sourceModel().removeChild(position, index.parent())

    def add(self, text=None, index=QtCore.QModelIndex()):
        """
        Custom: add node(s) under the specified index

        :param text: str. input text for de-serialization
        :param index: QModelIndex. parent index
        """
        # populate items with a temp root
        # jsonDict = ast.literal_eval(text)
        jsonDict = json.loads(text)
        root = QJsonNode.load(jsonDict)

        self.model().sourceModel().addChildren(root.children, index)
        self.model().sort(0, QtCore.Qt.AscendingOrder)

    def clear(self):
        """
        Custom: clear the entire view
        """
        self.model().sourceModel().clear()

    def copy(self):
        """
        Custom: copy the selected indices by store the serialized value
        """
        selected = self.asDict(self.getSelectedIndices())
        self._clipBroad = str(selected)

    def paste(self, index):
        """
        Custom: paste to index by de-serialize clipboard value

        :param index: QModelIndex. target index
        """
        self.customAdd(self._clipBroad, index)
        self._clipBroad = ''

    def customAdd(self, text=None, index=QtCore.QModelIndex()):
        """
        Custom: add node(s) under the specified index using specified values

        :param text: str. input text for de-serialization
        :param index: QModelIndex. parent index
        """

        # test value
        if not text:
            text = "{'key': 'value'}"

        dialog = QTextEditDialog(text)
        if dialog.exec_():
            text = dialog.getTextEdit()
            self.add(text, index)


class QTextEditDialog(QtWidgets.QDialog):
    """
    Custom pop-up dialog for editing text purpose, or getting long text input
    """

    def __init__(self, text='', title=''):
        """
        Initializing the dialog ui elements and connect signals

        :param text: str. pre-displayed text
        :param title: str. dialog title
        """
        super(QTextEditDialog, self).__init__()

        self.setWindowTitle(title)
        self.ui_textEdit = QtWidgets.QPlainTextEdit(text)
        self.ui_textEdit.setTabStopWidth(self.ui_textEdit.fontMetrics().width(' ') * 4)
        self.ui_acceptButton = QtWidgets.QPushButton("Confirm")

        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.ui_textEdit, 0, 0)
        layout.addWidget(self.ui_acceptButton, 1, 0)

        self.setLayout(layout)
        self.ui_acceptButton.clicked.connect(self.onClickAccept)

    def onClickAccept(self):
        """
        Trigger accept event when clicking the confirm button
        """
        if self.ui_textEdit.toPlainText():
            self.accept()
        else:
            print('value cannot be empty')

    def getTextEdit(self):
        """
        Get the text from text edit field

        :return: str. text from text edit field
        """
        return self.ui_textEdit.toPlainText()

    def closeEvent(self, event):
        """
        Overwrite the close event as it handles accept by default
        """
        self.close()


class QJsonNode(object):
    def __init__(self, parent=None):
        """
        Initialization

        :param parent: QJsonNode. parent of the current node
        """
        self._key = ""
        self._value = ""
        self._dtype = None
        self._parent = parent
        self._children = list()

    @classmethod
    def load(cls, value, parent=None):
        """
        Generate the hierarchical node tree using dictionary

        :param value: dict. input dictionary
        :param parent: QJsonNode. for recursive use only
        :return: QJsonNode. the top node
        """
        rootNode = cls(parent)
        rootNode.key = "root"
        rootNode.dtype = type(value)

        if isinstance(value, dict):
            # TODO: not sort will break things, but why?
            nodes = sorted(value.items())

            for key, value in nodes:
                child = cls.load(value, rootNode)
                child.key = key
                child.dtype = type(value)
                rootNode.addChild(child)
        elif isinstance(value, list):
            for index, value in enumerate(value):
                child = cls.load(value, rootNode)
                child.key = 'list[{}]'.format(index)
                child.dtype = type(value)
                rootNode.addChild(child)
        else:
            rootNode.value = value

        return rootNode

    @property
    def key(self):
        """
        Get key of the current node
        """
        return self._key

    @key.setter
    def key(self, key):
        self._key = key

    @property
    def value(self):
        """
        Get value of the current node
        """
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    @property
    def dtype(self):
        """
        Get value data type of the current node
        """
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        self._dtype = dtype

    @property
    def parent(self):
        """
        Get parent node of the current node
        """
        return self._parent

    @property
    def children(self):
        """
        Get the children of the current node
        :return: list.
        """
        return self._children

    @property
    def childCount(self):
        """
        Get the number of children of the current node
        :return: int.
        """
        return len(self._children)

    def addChild(self, node):
        """
        Add a new child to the current node

        :param node: QJsonNode. child node
        """
        self._children.append(node)
        node._parent = self

    def removeChild(self, position):
        """
        Remove child on row/position of the current node

        :param position: int. index of the children
        """
        node = self._children.pop(position)
        node._parent = None

    def child(self, row):
        """
        Get the child on row/position of the current node

        :param row: int. index of the children
        :return: QJsonNode. child node
        """
        return self._children[row]

    def row(self):
        """
        Get the current node's row/position in regards to its parent

        :return: int. index of the current node
        """
        if self._parent:
            return self.parent.children.index(self)
        return 0

    def asDict(self):
        """
        Serialize the hierarchical structure of current node to a dictionary

        :return: dict. serialization of the hierarchy
        """
        return {self.key: self.getChildrenValue(self)}

    def getChildrenValue(self, node):
        """
        Query the nested children value (instead of a single value)

        :param node: QJsonNode. root node
        :return: mixed. value
        """
        if node.dtype is dict:
            output = dict()
            for child in node.children:
                output[child.key] = self.getChildrenValue(child)
            return output
        elif node.dtype == list:
            output = list()
            for child in node.children:
                output.append(self.getChildrenValue(child))
            return output
        else:
            return node.value


class QJsonModel(QtCore.QAbstractItemModel):
    sortRole = QtCore.Qt.UserRole
    filterRole = QtCore.Qt.UserRole + 1

    def __init__(self, root, parent=None):
        """
        Initialization

        :param root: QJsonNode. root node of the model, it is hidden
        """
        super(QJsonModel, self).__init__(parent)
        self._rootNode = root

    def rowCount(self, parent=QtCore.QModelIndex()):
        """
        Override
        """
        if not parent.isValid():
            parentNode = self._rootNode
        else:
            parentNode = parent.internalPointer()

        return parentNode.childCount

    def columnCount(self, parent=QtCore.QModelIndex()):
        """
        Override
        """
        return 2

    def data(self, index, role):
        """
        Override
        """
        node = self.getNode(index)

        if role == QtCore.Qt.DisplayRole:
            if index.column() == 0:
                return node.key
            elif index.column() == 1:
                return node.value

        elif role == QtCore.Qt.EditRole:
            if index.column() == 0:
                return node.key
            elif index.column() == 1:
                return node.value

        elif role == QJsonModel.sortRole:
            return node.key

        elif role == QJsonModel.filterRole:
            return node.key

        elif role == QtCore.Qt.SizeHintRole:
            return QtCore.QSize(-1, 22)

    def setData(self, index, value, role):
        """
        Override
        """
        node = self.getNode(index)

        if role == QtCore.Qt.EditRole:
            if index.column() == 0:
                node.key = value
                self.dataChanged.emit(index, index)
                return True

            if index.column() == 1:
                node.value = value
                self.dataChanged.emit(index, index)
                return True

        return False

    def headerData(self, section, orientation, role):
        """
        Override
        """
        if role == QtCore.Qt.DisplayRole:
            if section == 0:
                return "KEY"
            elif section == 1:
                return "VALUE"

    def flags(self, index):
        """
        Override
        """
        flags = super(QJsonModel, self).flags(index)
        return (QtCore.Qt.ItemIsEditable
                | QtCore.Qt.ItemIsDragEnabled
                | QtCore.Qt.ItemIsDropEnabled
                | flags)

    def index(self, row, column, parent=QtCore.QModelIndex()):
        """
        Override
        """
        if not self.hasIndex(row, column, parent):
            return QtCore.QModelIndex()

        parentNode = self.getNode(parent)
        currentNode = parentNode.child(row)
        if currentNode:
            return self.createIndex(row, column, currentNode)
        else:
            return QtCore.QModelIndex()

    def parent(self, index):
        """
        Override
        """
        currentNode = self.getNode(index)
        parentNode = currentNode.parent

        if parentNode == self._rootNode:
            return QtCore.QModelIndex()

        return self.createIndex(parentNode.row(), 0, parentNode)

    def addChildren(self, children, parent=QtCore.QModelIndex()):
        """
        Custom: add children QJsonNode to the specified index
        """
        self.beginInsertRows(parent, 0, len(children) - 1)

        if parent == QtCore.QModelIndex():
            parentNode = self._rootNode
        else:
            parentNode = parent.internalPointer()

        for child in children:
            parentNode.addChild(child)

        self.endInsertRows()
        return True

    def removeChild(self, position, parent=QtCore.QModelIndex()):
        """
        Custom: remove child of position for the specified index
        """
        self.beginRemoveRows(parent, position, position)

        if parent == QtCore.QModelIndex():
            parentNode = self._rootNode
        else:
            parentNode = parent.internalPointer()

        parentNode.removeChild(position)

        self.endRemoveRows()
        return True

    def clear(self):
        """
        Custom: clear the model data
        """
        self.beginResetModel()
        self._rootNode = QJsonNode()
        self.endResetModel()
        return True

    def getNode(self, index):
        """
        Custom: get QJsonNode from model index

        :param index: QModelIndex. specified index
        """
        if index.isValid():
            currentNode = index.internalPointer()
            if currentNode:
                return currentNode
        return self._rootNode

    def asDict(self, index=QtCore.QModelIndex()):
        """
        Custom: serialize specified index to dictionary
        if no index is specified, the whole model will be serialized
        but will not include the root key (as it's supposed to be hidden)

        :param index: QModelIndex. specified index
        :return: dict. output dictionary
        """
        node = self.getNode(index)
        if node == self._rootNode:
            # print(node.asDict().values())
            return list(node.asDict().values())[0]

        return node.asDict()


class HighlightRule(object):
    def __init__(self, pattern, cformat):
        self.pattern = pattern
        self.format = cformat


class JsonHighlighter(QtGui.QSyntaxHighlighter):
    def __init__(self, parent=None):
        """
        Initialize rules with expression pattern and text format
        """
        super(JsonHighlighter, self).__init__(parent)
        self.rules = list()
        # numeric value
        cformat = QtGui.QTextCharFormat()
        cformat.setForeground(QtCore.Qt.darkBlue)
        cformat.setFontWeight(QtGui.QFont.Bold)
        pattern = QtCore.QRegExp("([-0-9.]+)(?!([^\"]*\"[\\s]*\\:))")

        rule = HighlightRule(pattern, cformat)
        self.rules.append(rule)

        # key
        cformat = QtGui.QTextCharFormat()
        pattern = QtCore.QRegExp("(\"[^\"]*\")\\s*\\:")
        # cformat.setForeground(QtCore.Qt.darkMagenta)
        cformat.setFontWeight(QtGui.QFont.Bold)

        rule = HighlightRule(pattern, cformat)
        self.rules.append(rule)

        # value
        cformat = QtGui.QTextCharFormat()
        pattern = QtCore.QRegExp(":+(?:[: []*)(\"[^\"]*\")")
        cformat.setForeground(QtCore.Qt.darkGreen)

        rule = HighlightRule(pattern, cformat)
        self.rules.append(rule)

    def highlightBlock(self, text):
        """
        Override: implementing virtual method of highlighting the text block
        """
        for rule in self.rules:
            # create a regular expression from the retrieved pattern
            expression = QtCore.QRegExp(rule.pattern)

            # check what index that expression occurs at with the ENTIRE text
            index = expression.indexIn(text)
            while index >= 0:
                # get the length of how long the expression is
                # set format from the start to the length with the text format
                length = expression.matchedLength()
                self.setFormat(index, length, rule.format)

                # set index to where the expression ends in the text
                index = expression.indexIn(text, index + length)

# if __name__ == "__main__":
#     #     jsonStr = '''
#     # '''
#     #     dct = json.loads(jsonStr)
#     #     print(dct)
#
#     import sys
#     from loguru import logger
#     import traceback
#     from PyQt5.QtWidgets import QMessageBox
#     from jsonEdit import QJsonEdit
#     from PyQt5.QtWidgets import QApplication
#
#
#     def exception_handler(exctype, value, tb):
#         """ Global exception handler for uncaught exceptions. """
#         logger.warning(value)
#         logger.warning(traceback.format_exc())
#         error_message = "".join(traceback.format_exception(exctype, value, tb))
#         QMessageBox.warning(
#             None, f"Exception: {exctype}",
#             f"{error_message}",
#             QMessageBox.Ok
#         )
#         sys.exit(1)
#
#
#     # Install global exception handler
#     sys.excepthook = exception_handler
#
#     app = QApplication(sys.argv)
#     widget = QJsonEdit()
#     widget.show()
#     sys.exit(app.exec_())
