import json
import os
import os
import sys
import traceback
from pathlib import Path

from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSlot, Qt, QThread, pyqtSignal, QUrl, QObject
from PyQt5.QtGui import QIcon, QDesktopServices
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QProgressDialog, QMessageBox
from bibidataprofile.code_editor import PythonHighlighter
from loguru import logger

from bibidataprofile.dataprofile import read_data_file, gen_data_profile, get_data_types, guess_y_factors, \
    binning_analysis_plot, fit_regression, fit_classification
# from bibidataprofile.mainwindow import Ui_MainWindow
# from bibidataprofile.mainwindowtabs import Ui_MainWindow
from bibidataprofile.dataprofileuitabs import Ui_MainWindow
from bibidataprofile.jsonEdit import QJsonEdit

bibipdf_home = Path(os.getenv('LOCALAPPDATA')) / '.bibidataprofile'

if not (bibipdf_home).exists():
    bibipdf_home.mkdir(exist_ok=True, parents=True)
logfile = bibipdf_home / 'bibidataprofile_{time:YYYY-MM-DD}.log'
logger.add(str(logfile.resolve()), rotation='1MB',
           level='DEBUG',
           encoding='utf-8',
           backtrace=True,
           diagnose=True)

COMMENTS_TAG = '#@!!@#'


class DataProfileWorker(QThread):
    finished = pyqtSignal()

    def __init__(self, datafile, workdir):
        super().__init__()
        self.datafile = datafile
        self.workdir = workdir
        self.data_df = None
        self.dataprofile_file = None
        self.data_types = None
        self.error_infos = {}

    def run(self):
        try:
            logger.info('DataProfileWorker Start ...')
            self.data_df = read_data_file(self.datafile)
            logger.info(f'data_df [{self.data_df.shape}]')
            self.dataprofile_file = gen_data_profile(self.data_df, self.workdir)
            logger.info(f'dataprofile_file [{self.dataprofile_file}]')
            self.data_types = get_data_types(self.data_df)
            logger.info(f'data_types [{self.data_types}]')
        except Exception as ex:
            logger.warning(ex)
            logger.warning(traceback.format_exc())
            self.error_infos['exception'] = ex
            self.error_infos['traceback'] = traceback.format_exc()

        self.finished.emit()


class VariableProfileWorker(QThread):
    finished = pyqtSignal()

    def __init__(self, data_df, type_variables, x_factors, y_factor, estimator, workdir):
        super().__init__()
        self.data_df = data_df
        self.x_factors = x_factors
        self.y_factor = y_factor
        self.estimator = estimator
        self.workdir = workdir
        self.type_variables = type_variables
        self.error_infos = {}

    def run(self):
        try:
            logger.info('VariableProfileWorker Start ...')
            logger.debug(f'data_df = {self.data_df}')
            logger.debug(f'x_factors = {self.x_factors}')
            logger.debug(f'y_factor = {self.y_factor}')
            logger.debug(f'type_variables = {self.type_variables}')
            if self.estimator == 'Regression':
                xlsx_path = (Path(self.workdir) / 'variable_profile.xlsx').resolve()
                fit_regression(
                    self.data_df[self.x_factors + [self.y_factor]],
                    self.x_factors,
                    self.y_factor,
                    xlsx_path=str(xlsx_path),
                    type_variables=self.type_variables,
                    verbose=True
                )
                logger.info(f'xlsx_path[exist={xlsx_path.exists()}]:{xlsx_path}')

                html_path = (Path(self.workdir) / "binning_analysis.html").resolve()
                binning_analysis_plot(
                    self.data_df[self.x_factors],
                    self.data_df[self.y_factor],
                    estimator_type='Regression',
                    type_variables=self.type_variables,
                    html_path=str(html_path)
                )
                logger.info(f'html_path[exist={html_path.exists()}]:{html_path}')

            elif self.estimator == 'Classification':
                xlsx_path = (Path(self.workdir) / 'variable_profile.xlsx').resolve()
                fit_classification(
                    self.data_df[self.x_factors + [self.y_factor]],
                    self.x_factors,
                    self.y_factor,
                    xlsx_path=str(xlsx_path),
                    type_variables=self.type_variables,
                    verbose=True
                )
                logger.info(f'xlsx_path[exist={xlsx_path.exists()}]:{xlsx_path}')

                html_path = (Path(self.workdir) / "binning_analysis.html").resolve()
                binning_analysis_plot(
                    self.data_df[self.x_factors],
                    self.data_df[self.y_factor],
                    estimator_type='Classification',
                    type_variables=self.type_variables,
                    html_path=str(html_path)
                )
                logger.info(f'html_path[exist={html_path.exists()}]:{html_path}')

        except Exception as ex:
            logger.warning(ex)
            logger.warning(traceback.format_exc())
            self.error_infos['exception'] = ex
            self.error_infos['traceback'] = traceback.format_exc()

        self.finished.emit()


def data_types_to_y_factors(data_types):
    yFactors = data_types.get('Numeric', []) + data_types.get('Categorical', [])
    return yFactors


def data_types_to_x_factors(data_types, yFactor):
    includes = data_types.get('Numeric', []) + data_types.get('Text', []) + data_types.get('Categorical', [])
    xFactors = [item for item in includes if item != yFactor]
    return xFactors


class BibiDataProfile(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # self.setWindowFlags(MainWindow.windowFlags() & ~ Qt.WindowType.WindowMaximizeButtonHint)
        # self.setWindowFlags(Qt.WindowType.WindowMinimizeButtonHint | Qt.WindowType.WindowCloseButtonHint)
        self.setWindowFlags(Qt.WindowType.WindowMinimizeButtonHint | Qt.WindowMinMaxButtonsHint
                            | Qt.WindowType.WindowCloseButtonHint)
        self.setWindowTitle('BibiDataProfile')
        self.setWindowIcon(QIcon(str(Path(__file__).parent / 'bibidataprofile.png')))
        # ''' self.plainTextEdit = QtWidgets.QPlainTextEdit(self.tab) --> self.plainTextEdit = CodeEditor(self.tab) '''
        self.highlighter = PythonHighlighter(self.plainTextEdit.document())
        self.plainTextEdit.setPlainText('''
def get_database_connection():
    conn = None
    return conn            
        ''')

        # ''' add JsonEdit dynamically '''
        self.jsonEdit = QJsonEdit(self.scrollAreaWidgetContents)
        self.jsonEdit.setObjectName("jsonEdit")
        self.gridLayout.addWidget(self.jsonEdit, 0, 0, 1, 1)

        self.jsonEdit_2 = QJsonEdit(self.scrollAreaWidgetContents_2)
        self.jsonEdit_2.setObjectName("jsonEdit_2")
        self.gridLayout_5.addWidget(self.jsonEdit_2, 0, 0, 1, 1)

        self.show()

        self.pushButton.clicked.connect(self.open_single_data_file)
        self.pushButton_2.clicked.connect(self.open_working_directory)
        self.pushButton_3.clicked.connect(self.analyse_data_profile)
        self.pushButton_4.clicked.connect(self.open_data_profile_directory)
        self.pushButton_5.clicked.connect(self.analyse_variable_profile)
        self.pushButton_6.clicked.connect(self.open_variable_profile_directory)

        self._data_df = None
        self._data_types = None

    @pyqtSlot()
    def analyse_variable_profile(self):
        try:
            if not (self.radioButton.isChecked() or self.radioButton_2.isChecked()):
                QMessageBox.warning(
                    self, "NO Estimator",
                    "Please select 'Regression' or 'Classification'",
                    QMessageBox.Ok
                )
                return
            elif self.radioButton.isChecked():
                estimator = 'Regression'
            else:
                estimator = 'Classification'

            workdir = self.lineEdit_2.text()
            y_factor = self.comboBox.currentText()
            data_types = self.jsonEdit.getDict()
            x_factors = self.jsonEdit_2.getDict()
            x_factors = [x_factor for x_factor in x_factors if x_factor != y_factor]
            self.jsonEdit_2.setDict(x_factors)
            logger.debug(f'x_factors = {x_factors}')
            logger.debug(f'y_factor = {y_factor}')
            if len(x_factors) < 1:
                QMessageBox.warning(
                    self, "NO X Factors",
                    f"x_factors={x_factors}",
                    QMessageBox.Ok
                )
                return
            if len(y_factor.strip()) == 0:
                QMessageBox.warning(
                    self, "NO Y Factor",
                    f"Please input y factor.",
                    QMessageBox.Ok
                )
                return
            if y_factor not in guess_y_factors(data_types):
                QMessageBox.warning(
                    self, "Y Factor NOT in Table",
                    f"y_factor={y_factor}, data_types={data_types}",
                    QMessageBox.Ok
                )
                return
            self.label_9.setText("variable profile linkage")
            self.label_9.setOpenExternalLinks(False)
            self.variableprofileworker = VariableProfileWorker(self._data_df, data_types, x_factors, y_factor,
                                                               estimator,
                                                               workdir)
            progress_dialog = QProgressDialog("Processing ...", "Close", 0, 0, self,
                                              Qt.WindowType.FramelessWindowHint)
            progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
            progress_dialog.setCancelButton(None)
            progress_dialog.show()
            self.variableprofileworker.finished.connect(progress_dialog.close)
            self.variableprofileworker.finished.connect(self.variable_profile_finished)
            self.pushButton_5.setEnabled(False)
            self.variableprofileworker.start()
            while self.variableprofileworker.isRunning():
                QApplication.processEvents()
            self.variableprofileworker.deleteLater()
        except Exception as ex:
            logger.warning(ex)
            logger.warning(traceback.format_exc())
            QMessageBox.warning(
                self, f"Exception: {ex}",
                f"{traceback.format_exc()}",
                QMessageBox.Ok
            )
            # QMessageBox.warning(
            #     self, f"Exception: {ex}",
            #     f"SEE <a href='{logfile.resolve().as_uri()}'>logfile for traceback.format_exc()</a>",
            #     QMessageBox.Ok
            # )

    def variable_profile_finished(self):
        self.pushButton_5.setEnabled(True)
        error_infos = self.variableprofileworker.error_infos
        if len(error_infos) > 0:
            QMessageBox.warning(
                self, f"Exception: {error_infos['exception']}",
                f"{error_infos['traceback']}",
                QMessageBox.Ok
            )
        variable_profile = Path(self.lineEdit_2.text().strip()) / 'variable_profile.xlsx'
        binning_analysis = Path(self.lineEdit_2.text().strip()) / 'binning_analysis.html'
        if variable_profile.exists():
            linkage = f"<a href={str((Path(self.lineEdit_2.text().strip()) / 'variable_profile.xlsx').resolve().as_uri())}>Variable Profile</a> \t\t\t\t "
        else:
            linkage = ""
        if binning_analysis.exists():
            linkage += f"<a href={str((Path(self.lineEdit_2.text().strip()) / 'binning_analysis.html').resolve().as_uri())}>Binning Analysis</a>  "
        else:
            linkage = ""
        if len(linkage) == 0:
            linkage = "variable profile linkage"
        logger.debug(f'linkage={linkage}')
        self.label_9.setText(linkage)
        self.label_9.setOpenExternalLinks(True)

    @pyqtSlot()
    def open_variable_profile_directory(self):
        try:
            workdir = self.lineEdit_2.text().strip()
            if Path(workdir).exists():
                QDesktopServices.openUrl(QUrl(
                    str(Path(workdir).as_uri())
                ))
            variable_profile = (Path(workdir) / 'variable_profile.xlsx').resolve()
            if variable_profile.exists():
                QDesktopServices.openUrl(QUrl(
                    str(variable_profile.as_uri())
                ))
            binning_analysis = (Path(workdir) / 'binning_analysis.html').resolve()
            if binning_analysis.exists():
                QDesktopServices.openUrl(QUrl(
                    str(binning_analysis.as_uri())
                ))
        except Exception as ex:
            logger.warning(ex)
            logger.warning(traceback.format_exc())
            QMessageBox.warning(
                self, f"Exception: {ex}",
                f"{traceback.format_exc()}",
                QMessageBox.Ok
            )
            # QMessageBox.warning(
            #     self, f"Exception: {ex}",
            #     f"SEE <a href='{logfile.resolve().as_uri()}'>logfile for traceback.format_exc()</a>",
            #     QMessageBox.Ok
            # )

    @pyqtSlot()
    def open_data_profile_directory(self):
        try:
            workdir = self.lineEdit_2.text().strip()
            data_profile = (Path(workdir) / 'data_profile.html').resolve()
            if data_profile.exists():
                QDesktopServices.openUrl(QUrl(
                    str(Path(workdir).as_uri())
                ))
                QDesktopServices.openUrl(QUrl(
                    str(data_profile.as_uri())
                ))
        except Exception as ex:
            logger.warning(ex)
            logger.warning(traceback.format_exc())
            QMessageBox.warning(
                self, f"Exception: {ex}",
                f"{traceback.format_exc()}",
                QMessageBox.Ok
            )
            # QMessageBox.warning(
            #     self, f"Exception: {ex}",
            #     f"SEE <a href='{logfile.resolve().as_uri()}'>logfile for traceback.format_exc()</a>",
            #     QMessageBox.Ok
            # )

    @pyqtSlot()
    def analyse_data_profile(self):
        try:
            datafile = self.lineEdit.text()
            workdir = self.lineEdit_2.text()
            if not Path(datafile).is_file() or not Path(datafile).exists():
                QMessageBox.warning(
                    self, "Data File NOT Found.",
                    f"datafile={datafile}",
                    QMessageBox.Ok
                )
                return
            if not Path(workdir).exists():
                QMessageBox.warning(
                    self, "Work Directory NOT Exist.",
                    f"workdir={workdir}",
                    QMessageBox.Ok
                )
                return

            self.label_4.setText("data profile linkage")
            self.label_4.setOpenExternalLinks(False)
            self.dataprofileworker = DataProfileWorker(datafile, workdir)
            progress_dialog = QProgressDialog("Processing ...", "Close", 0, 0, self,
                                              Qt.WindowType.FramelessWindowHint)
            progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
            progress_dialog.setCancelButton(None)
            progress_dialog.show()
            self.dataprofileworker.finished.connect(progress_dialog.close)
            self.dataprofileworker.finished.connect(self.data_profile_finished)
            self.dataprofileworker.start()
            self.pushButton_3.setEnabled(False)
            while self.dataprofileworker.isRunning():
                QApplication.processEvents()
            self.dataprofileworker.deleteLater()

        except Exception as ex:
            logger.warning(ex)
            logger.warning(traceback.format_exc())
            QMessageBox.warning(
                self, f"Exception: {ex}",
                f"{traceback.format_exc()}",
                QMessageBox.Ok
            )
            # QMessageBox.warning(
            #     self, f"Exception: {ex}",
            #     f"SEE <a href='{logfile.resolve().as_uri()}'>logfile for traceback.format_exc()</a>",
            #     QMessageBox.Ok
            # )

    def data_profile_finished(self):
        self.pushButton_3.setEnabled(True)
        error_infos = self.dataprofileworker.error_infos
        if len(error_infos) > 0:
            QMessageBox.warning(
                self, f"Exception: {error_infos['exception']}",
                f"{error_infos['traceback']}",
                QMessageBox.Ok
            )
        self._data_df = self.dataprofileworker.data_df
        self._data_types = self.dataprofileworker.data_types
        linkage = f"<a href={str((Path(self.lineEdit_2.text().strip()) / 'data_profile.html').resolve().as_uri())}>{self.label_4.text()}</a>"
        logger.debug(f'linkage={linkage}')
        self.label_4.setText(linkage)
        self.label_4.setOpenExternalLinks(True)
        # self.jsonEdit.setJson(
        #     data_types_to_json(self._data_types)
        # )
        self.jsonEdit.setDict(
            self._data_types
        )
        self.comboBox.addItems(data_types_to_y_factors(self._data_types))
        self.jsonEdit_2.setDict(
            data_types_to_x_factors(self._data_types, self.comboBox.currentText())
        )

    def batch_processing_finished(self):
        self.pushButton_6.setEnabled(True)

    @pyqtSlot()
    def open_single_data_file(self):
        try:
            filename, filetype = QFileDialog.getOpenFileName(
                self,
                "Open File",
                os.getcwd(),
                "All Files (*);; CSV Files (*.csv;*.tsv);;Excel Files (*.xlsx;*.xls);; HDF5 Files (*.h5)",
            )
            if Path(filename).is_file():
                self.lineEdit.setText(str(Path(filename).resolve()))
                if len(self.lineEdit_2.text().strip()) == 0 or not Path(self.lineEdit_2.text().strip()).exists():
                    self.lineEdit_2.setText(str((Path(filename).parent).resolve()))

        except Exception as ex:
            logger.warning(ex)
            logger.warning(traceback.format_exc())
            QMessageBox.warning(
                self, f"Exception: {ex}",
                f"{traceback.format_exc()}",
                QMessageBox.Ok
            )
            # QMessageBox.warning(
            #     self, f"Exception: {ex}",
            #     f"SEE <a href='{logfile.resolve().as_uri()}'>logfile for traceback.format_exc()</a>",
            #     QMessageBox.Ok
            # )

    @pyqtSlot()
    def open_working_directory(self):
        try:
            dirpath = QFileDialog.getExistingDirectory(
                self,
                caption="Open Directory",
                directory=os.getcwd(),
                options=QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks
            )
            if dirpath and Path(dirpath).exists():
                self.lineEdit_2.setText(str(Path(dirpath).resolve()))

        except Exception as ex:
            logger.warning(ex)
            logger.warning(traceback.format_exc())
            QMessageBox.warning(
                self, f"Exception: {ex}",
                f"{traceback.format_exc()}",
                QMessageBox.Ok
            )


class SafeApplication(QApplication):
    def notify(self, receiver, event):
        try:
            return super().notify(receiver, event)
        except Exception as ex:
            logger.warning(ex)
            logger.warning(traceback.format_exc())
            QMessageBox.warning(
                self, f"Exception: {ex}",
                f"{traceback.format_exc()}",
                QMessageBox.Ok
            )
            return False  # Propagate the exception


class ExceptionFilter(QObject):
    def eventFilter(self, obj, event):
        try:
            return super().eventFilter(obj, event)
        except Exception as ex:
            logger.warning(ex)
            logger.warning(traceback.format_exc())
            QMessageBox.warning(
                self, f"Exception: {ex}",
                f"{traceback.format_exc()}",
                QMessageBox.Ok
            )
            return True  # Prevent crash


def main():
    app = SafeApplication([])
    app.setStyleSheet("QMessageBox { messagebox-text-interaction-flags: 5; }")
    filter = ExceptionFilter()
    app.installEventFilter(filter)
    window = BibiDataProfile()
    sys.exit(app.exec())


def exception_handler(exctype, value, tb):
    """ Global exception handler for uncaught exceptions. """
    logger.warning(value)
    logger.warning(traceback.format_exc())
    error_message = "".join(traceback.format_exception(exctype, value, tb))
    QMessageBox.warning(
        None, f"Exception: {exctype}",
        f"{error_message}",
        QMessageBox.Ok
    )
    sys.exit(1)


# Install global exception handler
sys.excepthook = exception_handler

if __name__ == '__main__':
    main()
